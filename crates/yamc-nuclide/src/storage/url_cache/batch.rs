//! Downloading many nuclides' data in a few rounds of concurrent requests.
//!
//! See [`super::prefetch`] for why. The plan made here mirrors what the
//! download path will ask for, request by request: for a library keyword, the
//! release files the cache lacks and the byte ranges [`fetch_ranged`] asks for
//! (see `release_cache`), and for a raw URL what [`super::download_sections`]
//! asks for. The replies are held in [`PREFETCHED`] until it asks. Nothing
//! else changes: retries, the 404 handling, the multipart parsing, every
//! sha256 check against the release manifest and the range index, staging and
//! publishing are all the download path's own, run on bytes that were fetched
//! earlier.

use super::*;
use crate::storage::release_cache::{fetch_particle, Registry};

/// How many nuclides are planned and fetched together.
///
/// Every reply of a batch is held in memory until the batch is cached, so this
/// bounds that memory. In the browser it is wasm heap: a batch of activation
/// loads is tens of MB, a batch of transport loads at one temperature about the
/// same. Large enough that a round keeps a pool of fetchers busy, which is all
/// the speed there is to win.
const NUCLIDES_PER_BATCH: usize = 32;

/// Replies fetched ahead of the download that will ask for them, keyed by the
/// exact request.
static PREFETCHED: Lazy<Mutex<HashMap<Get, Result<Reply, String>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

fn prefetched() -> std::sync::MutexGuard<'static, HashMap<Get, Result<Reply, String>>> {
    PREFETCHED.lock().unwrap_or_else(|p| p.into_inner())
}

/// Every request a held reply was taken for, so the tests can tell a download
/// served from a prefetch from one that went to the network.
#[cfg(test)]
pub(crate) static TAKEN: Lazy<Mutex<Vec<Get>>> = Lazy::new(|| Mutex::new(Vec::new()));

/// The reply fetched ahead for `get`, if there is one. Taken rather than read,
/// so a retry of the same request goes to the network.
pub(super) fn take(get: &Get) -> Option<Result<Reply, String>> {
    let mut held = prefetched();
    if held.is_empty() {
        return None;
    }
    let reply = held.remove(get);
    #[cfg(test)]
    if reply.is_some() {
        TAKEN
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(get.clone());
    }
    reply
}

/// One nuclide the download path will fetch something for.
struct Plan<'a> {
    request: &'a DataRequest<'a>,
    /// The URL the sections are under: each is `<url>/<section>`.
    url: String,
    /// The folder the download caches into.
    dir: PathBuf,
    ranging: Ranged<'a>,
    /// The sections the download will fetch, whole or as byte ranges.
    sections: Vec<String>,
    /// Whether the download verifies what it fetches (a library keyword), and
    /// so ranges a section only when every entry it needs has a hash.
    verified: bool,
}

impl Plan<'_> {
    fn url(&self, section: &str) -> String {
        format!("{}/{}", self.url, section)
    }

    /// The GETs for `section`, as the download will make them, given the
    /// `version.json` it will plan from.
    ///
    /// On the keyword path that `version.json` is one the download verifies
    /// before it ranges anything: one that does not match the manifest fails
    /// the download first, and what was planned from it is dropped unused.
    fn gets(&self, section: &str, version: Option<&serde_json::Value>) -> Vec<Get> {
        let url = self.url(section);
        let whole = || vec![(url.clone(), None)];
        let Some(version) = version else {
            return whole();
        };
        let plan = match plan_ranges(version, &self.dir, section, self.ranging, self.request.name) {
            Ok(Some(plan)) if !self.verified || plan.verifiable() => plan,
            Ok(_) => return whole(),
            // The download fails the same way before fetching anything, and
            // says why.
            Err(_) => return Vec::new(),
        };
        plan.spans
            .chunks(MAX_RANGES_PER_REQUEST)
            .map(|chunk| (url.clone(), Some(range_header(chunk))))
            .collect()
    }
}

/// What the download path will fetch for `request`, or `None` when it will
/// fetch nothing (cached, not published, offline, or not a download at all).
///
/// For a keyword this resolves the library (its `latest.json` and manifest)
/// if nothing in the process has yet, as the download would first.
fn plan<'a>(registry: &Registry, request: &'a DataRequest<'a>) -> Option<Plan<'a>> {
    let ranging = ranged(request.kind, request.scope);
    if is_keyword(request.source) {
        let pending =
            registry.pending(request.source, request.name, request.kind, request.scope)?;
        return Some(Plan {
            request,
            url: pending.url,
            dir: pending.local,
            ranging,
            sections: pending.sections,
            verified: true,
        });
    }
    if !is_url(request.source) {
        return None;
    }
    let dir = get_cache_dir()
        .ok()?
        .join(format!("{}.arrow", request.name));
    let sections = sections_for(request.kind, request.scope);
    if have_all_sections(&dir, &sections, ranging) {
        return None;
    }
    Some(Plan {
        request,
        url: request.source.to_string(),
        sections: sections_to_fetch(&dir, &sections, ranging)
            .into_iter()
            .map(|(name, _)| name.to_string())
            .collect(),
        dir,
        ranging,
        verified: false,
    })
}

/// Fetch every one of `gets` at once and hold the replies for the download.
fn fill(gets: &mut Vec<Get>) {
    let mut seen = HashSet::new();
    gets.retain(|get| seen.insert(get.clone()));
    if gets.is_empty() {
        return;
    }
    let replies = send_many(gets);
    prefetched().extend(gets.iter().cloned().zip(replies));
}

/// The parsed body of a held `version.json` reply, if it was a success.
fn held_version(url: &str) -> Option<serde_json::Value> {
    let held = prefetched();
    match held.get(&(url.to_string(), None))? {
        Ok(Reply {
            status: 200..=299,
            body: Ok(bytes),
        }) => serde_json::from_slice(bytes).ok(),
        _ => None,
    }
}

std::thread_local! {
    /// Set while a prefetch caches its nuclides, whose progress it reports
    /// itself (see [`Progress`]).
    static QUIET: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether a download should leave its per-nuclide line out.
pub(crate) fn quiet() -> bool {
    QUIET.with(|q| q.get())
}

/// The `Downloaded 4 of 9 nuclides` line a prefetch prints as it goes.
///
/// A count rather than a line per nuclide naming its files, which a
/// transmutation closure of hundreds of nuclides turned into a flood. Each
/// line starts the same way, so a front end can show only the latest (the
/// browser page does, as it does for transport's `Progress:` lines). A line is
/// printed at the start, so the count is known before anything arrives, then
/// at most every [`Progress::EVERY`] and at the end. The nuclides of a batch
/// finish together, once its requests are answered, so a line for each would
/// only repeat itself.
struct Progress {
    done: usize,
    total: usize,
    what: &'static str,
    printed: std::time::Instant,
}

impl Progress {
    const EVERY: std::time::Duration = std::time::Duration::from_millis(250);

    fn start(plans: &[Plan]) -> Self {
        let photons = plans
            .iter()
            .filter(|p| p.request.kind == DataKind::Photon)
            .count();
        let what = match (photons, plans.len() - photons) {
            (0, _) => "nuclides",
            (_, 0) => "elements",
            _ => "nuclides and elements",
        };
        let progress = Self {
            done: 0,
            total: plans.len(),
            what,
            printed: std::time::Instant::now(),
        };
        progress.print();
        progress
    }

    fn print(&self) {
        use std::io::Write;
        let mut out = std::io::stdout().lock();
        let _ = writeln!(
            out,
            "Downloaded {} of {} {}",
            self.done, self.total, self.what
        );
        let _ = out.flush();
    }

    fn one_done(&mut self) {
        self.done += 1;
        if self.done == self.total || self.printed.elapsed() >= Self::EVERY {
            self.print();
            self.printed = std::time::Instant::now();
        }
    }
}

/// [`super::prefetch`], with library keywords resolved through `registry`.
pub(crate) fn prefetch(registry: &Registry, requests: &[DataRequest<'_>]) {
    if client().is_err() {
        return;
    }
    // One plan per cache directory: a second request for the same nuclide at
    // another scope is a top-up the download makes after the first, from what
    // the first wrote, so it is left to the download. Planning resolves every
    // keyword the requests name, so the pointer and manifest of each are
    // fetched once, before any batch.
    let mut dirs = HashSet::new();
    let plans: Vec<Plan> = requests
        .iter()
        .filter_map(|r| plan(registry, r))
        .filter(|p| dirs.insert(p.dir.clone()))
        .collect();
    if plans.is_empty() {
        return;
    }
    let mut progress = Progress::start(&plans);

    for batch in plans.chunks(NUCLIDES_PER_BATCH) {
        // Round one: everything the index does not decide, and the ranged
        // sections of nuclides whose index is already cached.
        let mut first = Vec::new();
        let mut waiting = Vec::new();
        for plan in batch {
            let fetches_version = plan.sections.iter().any(|s| s == "version.json");
            let cached = (!fetches_version)
                .then(|| cached_version(&plan.dir))
                .flatten();
            for section in &plan.sections {
                if !ranges_section(section, plan.ranging) {
                    first.push((plan.url(section), None));
                } else if fetches_version {
                    waiting.push((plan, section));
                } else {
                    first.extend(plan.gets(section, cached.as_ref()));
                }
            }
        }
        fill(&mut first);

        // Round two: the byte ranges the fetched indexes name, or the whole
        // section where those cannot be used. A nuclide whose version.json
        // failed plans nothing, and its download meets the failure again.
        let mut second = Vec::new();
        for (plan, section) in waiting {
            if let Some(version) = held_version(&plan.url("version.json")) {
                second.extend(plan.gets(section, Some(&version)));
            }
        }
        fill(&mut second);

        QUIET.with(|q| q.set(true));
        for plan in batch {
            let r = plan.request;
            if is_keyword(r.source) {
                let _ = fetch_particle(registry, r.source, r.name, r.kind, r.scope);
            } else {
                let _ = download_and_cache(r.source, r.source, r.name, r.kind, r.scope);
            }
            progress.one_done();
        }
        QUIET.with(|q| q.set(false));

        // Whatever the downloads did not ask for. Only this batch's, so a
        // concurrent prefetch on another thread keeps its own.
        let mut held = prefetched();
        for get in first.iter().chain(&second) {
            held.remove(get);
        }
    }
}
