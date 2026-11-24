always make sure cargo check, cargo fmt, cargo clippy and ruff check pass before finished the task and before making a pull request

don't use try excepts, it is better to fail with clear message

don't use rust unsafe.

thin python wrappers for the rust functions. Logic should be in the rust core.

Makes nuclear data easy with automatic download of arrow files but also local options available.

we don't need to be backwards compatible

don't use alias to rename things, apart from the standard Py prefix for python wrappers

always commit as github username shimwell email address mail@jshimwell.com

be concise

avoid unnecessary user arguments that can be automated away.

the focus of yamc is fusion fixed source simulations providing the suite of fusion neutronics analysis, easy and quick to get started with a simple install that works on all OS and main architectures, it should be accurate, fast, simple API that supports you when post processing. 

the focus of yani is an inventory code / activation / transmutations / depletion simulations from a neutron flux to get the evolved material at different time steps. Useful properties such as activity, contact dose, decay heat are provided to the user. Should use the similar nuclear data to yamc (continuous energy). Focus is accuracy, easy of install, ease of deployment and speed. WASM supported as well as the main OS and architectures