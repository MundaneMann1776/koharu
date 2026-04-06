use anyhow::{Result, bail};

use crate::Runtime;

pub(crate) fn package_enabled(_: &Runtime) -> bool {
    false
}

pub(crate) fn package_present(_: &Runtime) -> Result<bool> {
    Ok(false)
}

pub(crate) async fn package_prepare(_: &Runtime) -> Result<()> {
    Ok(())
}

pub(crate) async fn ensure_ready(_: &Runtime) -> Result<()> {
    bail!("ZLUDA is only supported on Windows")
}

pub(crate) fn backend_status(_: &Runtime) -> Result<()> {
    bail!("ZLUDA is only supported on Windows")
}

pub(crate) fn candidate_status(_: &Runtime) -> Result<()> {
    bail!("ZLUDA is only supported on Windows")
}

crate::declare_native_package!(
    id: "runtime:zluda",
    bootstrap: true,
    order: 11,
    enabled: crate::zluda::package_enabled,
    present: crate::zluda::package_present,
    prepare: crate::zluda::package_prepare,
);
