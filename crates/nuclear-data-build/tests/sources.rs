//! Getting the source files: the folders searched, matching by contents, the
//! host tried on every run, and the one message for files a browser must
//! fetch. Every case runs against a local origin.

mod common;

use common::*;
use nuclear_data_build::recipe::{Recipe, Sublibrary};
use nuclear_data_build::sources::{downloads_dir, fetch, Search};

const JAEA_STUB: &[u8] = b"<html><head><meta http-equiv=\"refresh\" content=\"0;URL=/\"></head></html>";
const CHALLENGE: &[u8] = b"<!DOCTYPE html><html><title>Just a moment...</title></html>";

fn body(tag: u8, n: usize) -> Vec<u8> {
    (0..n).map(|i| (i as u8).wrapping_mul(31) ^ tag).collect()
}

fn no_os_folder() -> Search {
    Search::default()
}

#[test]
fn a_renamed_copy_in_downloads_is_used_where_it_is_without_a_request() {
    let origin = Origin::start();
    let scratch = Scratch::new("in-downloads");
    let file = body(1, 5000);
    let url = origin.serve("jendl5-n.tar.gz", Reply::file(&file));
    let r = recipe("jendl-5.0", vec![source("jendl5-n.tar.gz", &url, &file, Sublibrary::Neutron)]);
    let shared = scratch.dir("shared");
    let renamed = shared.join("jendl5-n (1).tar.gz");
    std::fs::write(&renamed, &file).unwrap();
    // Same size, other contents: never taken for the file.
    let mut other = file.clone();
    other[0] ^= 1;
    std::fs::write(shared.join("jendl5-n.tar.gz"), &other).unwrap();

    let root = scratch.dir("lib");
    let search = Search {
        downloads: Some(shared.clone()),
        os_downloads: None,
    };
    let got = fetch(&r, &root, &search).unwrap();
    assert_eq!(got[0].path, renamed);
    assert_eq!(origin.hits("jendl5-n.tar.gz"), 0);
    assert!(!downloads_dir(&root).join("jendl5-n.tar.gz").exists());
}

#[test]
fn a_copy_in_the_os_downloads_folder_is_copied_into_the_build() {
    let origin = Origin::start();
    let scratch = Scratch::new("os-downloads");
    let file = body(2, 4000);
    let url = origin.serve("IRDFF-II_ENDF.zip", Reply::file(&file));
    let r = recipe("irdff-ii", vec![source("IRDFF-II_ENDF.zip", &url, &file, Sublibrary::Neutron)]);
    let os = scratch.dir("Téléchargements");
    std::fs::write(os.join("IRDFF-II_ENDF(2).zip"), &file).unwrap();
    // A browser's download still in progress, which is shorter.
    std::fs::write(os.join("IRDFF-II_ENDF.zip.part"), &file[..100]).unwrap();

    let root = scratch.dir("lib");
    let search = Search {
        downloads: None,
        os_downloads: Some(os.clone()),
    };
    let got = fetch(&r, &root, &search).unwrap();
    let dest = downloads_dir(&root).join("IRDFF-II_ENDF.zip");
    assert_eq!(got[0].path, dest);
    assert_eq!(read(&dest), file);
    assert_eq!(origin.hits("IRDFF-II_ENDF.zip"), 0);
    // The user's copy is left alone.
    assert!(os.join("IRDFF-II_ENDF(2).zip").exists());
}

#[test]
fn the_host_is_asked_and_the_next_run_reuses_the_file() {
    let origin = Origin::start();
    let scratch = Scratch::new("host");
    let file = body(3, 6000);
    let url = origin.serve("decay.tar.gz", Reply::file(&file));
    let r = recipe("endf-b8.1", vec![source("decay.tar.gz", &url, &file, Sublibrary::Decay)]);
    let root = scratch.dir("lib");
    for _ in 0..2 {
        let got = fetch(&r, &root, &no_os_folder()).unwrap();
        assert_eq!(read(&got[0].path), file);
    }
    assert_eq!(origin.hits("decay.tar.gz"), 1);
}

#[test]
fn the_host_is_tried_before_asking_for_a_browser() {
    let origin = Origin::start();
    let scratch = Scratch::new("host-first");
    let file = body(4, 3000);
    // A host that refused scripts on another day and serves the file today.
    let url = origin.serve("fendl-3.2d-T1-neutron-endf.zip", Reply::file(&file));
    let r = recipe(
        "fendl-3.2d",
        vec![source("fendl-3.2d-T1-neutron-endf.zip", &url, &file, Sublibrary::Neutron)],
    );
    let root = scratch.dir("lib");
    fetch(&r, &root, &no_os_folder()).unwrap();
    assert_eq!(origin.hits("fendl-3.2d-T1-neutron-endf.zip"), 1);
}

/// Every file is tried, the ones that download are kept, and one message
/// lists every file a browser must fetch with its link and size, then
/// anything that failed another way.
#[test]
fn one_message_lists_every_file_a_browser_must_fetch() {
    let origin = Origin::start();
    let scratch = Scratch::new("message");
    let ok = body(5, 2000);
    let blocked = body(6, 2_200_000);
    let stub = body(7, 19_000);
    let flare = body(8, 3000);
    let gone = body(9, 1000);
    let ok_url = origin.serve("ok.tar.gz", Reply::file(&ok));
    let blocked_url = origin.serve(
        "IRDFF/IRDFF-II_ENDF.zip",
        Reply {
            status: "403 Forbidden",
            headers: "Content-Type: text/html\r\n",
            body: CHALLENGE.to_vec(),
        },
    );
    let stub_url = origin.serve(
        "JENDL/jendl5-n.tar.gz",
        Reply {
            status: "200 OK",
            headers: "Content-Type: text/html\r\n",
            body: JAEA_STUB.to_vec(),
        },
    );
    let flare_url = origin.serve(
        "fendl/atom.zip",
        Reply {
            status: "503 Service Unavailable",
            headers: "Content-Type: text/html\r\ncf-mitigated: challenge\r\n",
            body: CHALLENGE.to_vec(),
        },
    );
    let gone_url = format!("{}moved.zip", origin.url);
    let r: Recipe = recipe(
        "mixed",
        vec![
            source("ok.tar.gz", &ok_url, &ok, Sublibrary::Neutron),
            source("IRDFF-II_ENDF.zip", &blocked_url, &blocked, Sublibrary::Neutron),
            source("jendl5-n.tar.gz", &stub_url, &stub, Sublibrary::Neutron),
            source("atom.zip", &flare_url, &flare, Sublibrary::Photoatomic),
            source("moved.zip", &gone_url, &gone, Sublibrary::Decay),
        ],
    );
    let root = scratch.dir("lib");
    let os = scratch.dir("Downloads");
    let search = Search {
        downloads: None,
        os_downloads: Some(os.clone()),
    };
    let err = fetch(&r, &root, &search).unwrap_err();

    assert!(
        err.starts_with("mixed needs 3 files that their hosts only serve to a web browser."),
        "{err}"
    );
    for url in [&blocked_url, &stub_url, &flare_url] {
        assert!(err.contains(url.as_str()), "{url} missing from:\n{err}");
    }
    assert!(err.contains("2 MB"), "{err}");
    assert!(err.contains(&os.display().to_string()), "{err}");
    assert!(err.contains(&downloads_dir(&root).display().to_string()), "{err}");
    assert!(err.contains("Then run the build again."), "{err}");
    assert!(!err.contains("no Downloads folder"), "{err}");
    // The 404 is not a browser's job, and is reported on its own.
    assert!(err.contains("these source files could not be downloaded"), "{err}");
    assert!(err.contains("HTTP 404"), "{err}");
    assert!(!err.contains(&format!("  {ok_url}")), "{err}");
    // The file that downloaded is kept, and the challenge was not retried.
    assert_eq!(read(&downloads_dir(&root).join("ok.tar.gz")), ok);
    assert_eq!(origin.hits("fendl/atom.zip"), 1);

    // Saved by a browser under its own names, the next run finds them all
    // and asks for nothing but the 404.
    std::fs::write(os.join("IRDFF-II_ENDF (1).zip"), &blocked).unwrap();
    std::fs::write(os.join("jendl5-n.tar(1).gz"), &stub).unwrap();
    std::fs::write(os.join("atom.zip"), &flare).unwrap();
    let err = fetch(&r, &root, &search).unwrap_err();
    assert!(err.starts_with("mixed: these source files could not be downloaded"), "{err}");
    assert_eq!(origin.hits("ok.tar.gz"), 1);
    assert_eq!(origin.hits("IRDFF/IRDFF-II_ENDF.zip"), 1);
}

#[test]
fn with_no_downloads_folder_the_message_asks_for_one() {
    let origin = Origin::start();
    let scratch = Scratch::new("headless");
    let file = body(10, 1000);
    let url = origin.serve(
        "x.zip",
        Reply {
            status: "403 Forbidden",
            headers: "Content-Type: text/html\r\n",
            body: CHALLENGE.to_vec(),
        },
    );
    let r = recipe("fendl-3.2d", vec![source("x.zip", &url, &file, Sublibrary::Neutron)]);
    let root = scratch.dir("lib");
    let err = fetch(&r, &root, &no_os_folder()).unwrap_err();
    assert!(
        err.contains("This system has no Downloads folder, so save them to a folder and pass it as downloads="),
        "{err}"
    );
    assert!(
        err.starts_with("fendl-3.2d needs 1 file that its host only serves to a web browser."),
        "{err}"
    );
}

#[test]
fn a_recipe_with_an_unpinned_source_is_refused_before_any_request() {
    let origin = Origin::start();
    let scratch = Scratch::new("unpinned");
    let file = body(11, 1000);
    let url = origin.serve("a.tar.gz", Reply::file(&file));
    let mut unpinned = source("b.tar.gz", &url, &file, Sublibrary::Neutron);
    unpinned.pin = None;
    let r = recipe(
        "jendl-5.0",
        vec![source("a.tar.gz", &url, &file, Sublibrary::Neutron), unpinned],
    );
    let err = fetch(&r, &scratch.dir("lib"), &no_os_folder()).unwrap_err();
    assert!(err.contains("jendl-5.0 cannot be built yet"), "{err}");
    assert!(err.contains("b.tar.gz"), "{err}");
    assert_eq!(origin.hits("a.tar.gz"), 0);
}

#[test]
fn the_real_recipes_not_pinned_yet_are_refused() {
    let scratch = Scratch::new("real-unpinned");
    let r = nuclear_data_build::recipe::recipe("jendl-5.0").unwrap();
    let err = fetch(r, &scratch.dir("lib"), &no_os_folder()).unwrap_err();
    assert!(err.contains("jendl-5.0 cannot be built yet"), "{err}");
}
