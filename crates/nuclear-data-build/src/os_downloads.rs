//! The operating system's own Downloads folder, where a web browser saves the
//! archives some hosts only serve to a browser.
//!
//! Guessing `~/Downloads` is not enough. Linux desktops name the folder in the
//! user's language (`~/Téléchargements`) and record it as `XDG_DOWNLOAD_DIR`,
//! and Windows users can move the Downloads known folder to another drive.

#[cfg(not(any(windows, target_os = "macos")))]
use std::path::Path;
use std::path::PathBuf;

/// The Downloads folder, or `None` when the system has none (a headless
/// machine with no home folder, for example).
pub fn download_dir() -> Option<PathBuf> {
    platform::download_dir()
}

/// `XDG_DOWNLOAD_DIR` from the text of a `user-dirs.dirs` file, resolved
/// against `home`.
///
/// The file holds shell assignments such as
/// `XDG_DOWNLOAD_DIR="$HOME/Téléchargements"`. A value is either `$HOME/...`
/// or an absolute path; anything else is not valid there and is ignored.
/// `"$HOME/"` on its own is how a user turns the folder off, so downloads then
/// land in the home folder itself.
#[cfg(not(any(windows, target_os = "macos")))]
pub fn parse_user_dirs(text: &str, home: &Path) -> Option<PathBuf> {
    let value = text.lines().find_map(|line| {
        let line = line.trim();
        let rest = line.strip_prefix("XDG_DOWNLOAD_DIR=")?;
        rest.strip_prefix('"')?.strip_suffix('"')
    })?;
    resolve_xdg_value(value, home)
}

#[cfg(not(any(windows, target_os = "macos")))]
fn resolve_xdg_value(value: &str, home: &Path) -> Option<PathBuf> {
    if value == "$HOME" {
        return Some(home.to_path_buf());
    }
    if let Some(rest) = value.strip_prefix("$HOME/") {
        let rest = rest.trim_end_matches('/');
        return Some(if rest.is_empty() {
            home.to_path_buf()
        } else {
            home.join(rest)
        });
    }
    let path = Path::new(value);
    path.is_absolute().then(|| path.to_path_buf())
}

#[cfg(windows)]
mod platform {
    use std::ffi::OsString;
    use std::os::windows::ffi::OsStringExt;
    use std::path::PathBuf;
    use std::{ptr, slice};

    use windows_sys::Win32::Foundation::S_OK;
    use windows_sys::Win32::System::Com::CoTaskMemFree;
    use windows_sys::Win32::UI::Shell::{
        FOLDERID_Downloads, SHGetKnownFolderPath, KF_FLAG_DONT_VERIFY,
    };

    unsafe extern "C" {
        fn wcslen(buf: *const u16) -> usize;
    }

    /// The Downloads known folder, wherever the user has moved it.
    pub fn download_dir() -> Option<PathBuf> {
        // SAFETY: SHGetKnownFolderPath writes a CoTaskMemAlloc'd,
        // NUL-terminated wide string into `path` on success, which is read
        // once and then freed. On failure `path` may still be allocated, and
        // freeing a null pointer is a no-op.
        unsafe {
            let mut path = ptr::null_mut();
            let hr = SHGetKnownFolderPath(
                &FOLDERID_Downloads,
                KF_FLAG_DONT_VERIFY as u32,
                ptr::null_mut(),
                &mut path,
            );
            let out = (hr == S_OK).then(|| {
                PathBuf::from(OsString::from_wide(slice::from_raw_parts(
                    path,
                    wcslen(path),
                )))
            });
            CoTaskMemFree(path.cast());
            out
        }
    }
}

#[cfg(target_os = "macos")]
mod platform {
    use std::path::PathBuf;

    /// `~/Downloads`, which macOS does not let users rename or move.
    pub fn download_dir() -> Option<PathBuf> {
        etcetera::home_dir().ok().map(|home| home.join("Downloads"))
    }
}

#[cfg(not(any(windows, target_os = "macos")))]
mod platform {
    use std::path::{Path, PathBuf};

    use super::{parse_user_dirs, resolve_xdg_value};

    /// `XDG_DOWNLOAD_DIR`, from the environment or from
    /// `$XDG_CONFIG_HOME/user-dirs.dirs`, as `xdg-user-dir DOWNLOAD` reads
    /// it; `~/Downloads` when neither sets it.
    pub fn download_dir() -> Option<PathBuf> {
        let home = etcetera::home_dir().ok()?;
        if let Some(dir) = std::env::var("XDG_DOWNLOAD_DIR")
            .ok()
            .and_then(|v| resolve_xdg_value(&v, &home))
        {
            return Some(dir);
        }
        let config = std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .filter(|p| p.is_absolute())
            .unwrap_or_else(|| home.join(".config"));
        std::fs::read_to_string(config.join("user-dirs.dirs"))
            .ok()
            .and_then(|text| parse_user_dirs(&text, &home))
            .or_else(|| Some(Path::new(&home).join("Downloads")))
    }
}

#[cfg(all(test, not(any(windows, target_os = "macos"))))]
mod tests {
    use super::*;

    const HOME: &str = "/home/user";

    #[test]
    fn reads_a_translated_folder() {
        let text = "# written by xdg-user-dirs-update\n\
                    XDG_DESKTOP_DIR=\"$HOME/Bureau\"\n\
                    XDG_DOWNLOAD_DIR=\"$HOME/Téléchargements\"\n";
        assert_eq!(
            parse_user_dirs(text, Path::new(HOME)),
            Some(PathBuf::from("/home/user/Téléchargements"))
        );
    }

    #[test]
    fn reads_an_absolute_folder() {
        let text = "XDG_DOWNLOAD_DIR=\"/data/downloads\"\n";
        assert_eq!(
            parse_user_dirs(text, Path::new(HOME)),
            Some(PathBuf::from("/data/downloads"))
        );
    }

    #[test]
    fn a_disabled_folder_is_the_home_folder() {
        let text = "XDG_DOWNLOAD_DIR=\"$HOME/\"\n";
        assert_eq!(
            parse_user_dirs(text, Path::new(HOME)),
            Some(PathBuf::from(HOME))
        );
    }

    #[test]
    fn ignores_comments_other_keys_and_relative_values() {
        let text = "#XDG_DOWNLOAD_DIR=\"$HOME/Old\"\nXDG_MUSIC_DIR=\"$HOME/Music\"\n";
        assert_eq!(parse_user_dirs(text, Path::new(HOME)), None);
        let text = "XDG_DOWNLOAD_DIR=\"Downloads\"\n";
        assert_eq!(parse_user_dirs(text, Path::new(HOME)), None);
    }
}
