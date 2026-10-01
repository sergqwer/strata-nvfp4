# tools/enable-large-pages.ps1 - let this Windows account use 2 MB "large pages" for Strata's expert arena.
#
#   powershell -ExecutionPolicy Bypass -File tools\enable-large-pages.ps1           grant (asks for admin via UAC)
#   powershell -ExecutionPolicy Bypass -File tools\enable-large-pages.ps1 -Check    is it active in this sign-in?
#   powershell -ExecutionPolicy Bypass -File tools\enable-large-pages.ps1 -Revoke   take it back
#
# It grants "Lock pages in memory" (SeLockMemoryPrivilege) to the current user - what secpol.msc > Local Policies >
# User Rights Assignment does, but through secedit, so it works on Windows Home too. Windows puts a privilege into
# a sign-in's token only when the sign-in starts: SIGN OUT AND BACK IN (or reboot) afterwards. Locking the screen
# is not enough. The policy as it was is saved to %LOCALAPPDATA%\strata-large-pages\before.inf.
param([switch]$Check, [switch]$Revoke, [string]$Sid)
$ErrorActionPreference = 'Stop'
$dir = Join-Path $env:LOCALAPPDATA 'strata-large-pages'

function Test-Active {
    # the full path: with Git's usr\bin on PATH a bare `whoami` is the MSYS one, which knows no /priv
    $line = (& (Join-Path $env:SystemRoot 'System32\whoami.exe') /priv) | Where-Object { $_ -match 'SeLockMemoryPrivilege' }
    return [bool]$line
}

if ($Check) {
    if (Test-Active) {
        Write-Output 'Large pages: ACTIVE for this sign-in (SeLockMemoryPrivilege is in the token; "Disabled" there is'
        Write-Output 'normal - Strata enables it itself). The engine log should say: expert arena ... large pages (2097152 B).'
    } else {
        Write-Output 'Large pages: NOT active in this sign-in. Run this script without -Check, then sign out and back in.'
    }
    exit 0
}

if (-not $Sid) { $Sid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value }
$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) {
    # the SID is taken here, before elevating: an elevation with another admin account must not grant that one
    $a = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$PSCommandPath`"", '-Sid', $Sid)
    if ($Revoke) { $a += '-Revoke' }
    $p = Start-Process -FilePath 'powershell.exe' -ArgumentList $a -Verb RunAs -Wait -PassThru
    if ($p.ExitCode -ne 0) { Write-Output "failed (exit $($p.ExitCode)); details in $dir\log.txt"; exit $p.ExitCode }
    if ($Revoke) { Write-Output 'Revoked. It stops working at the next sign-in.' }
    else { Write-Output 'Granted. Now SIGN OUT and back in (or reboot); then check with -Check.' }
    exit 0
}

New-Item -ItemType Directory -Force $dir | Out-Null
$log = Join-Path $dir 'log.txt'
function Log($m) { Add-Content $log ("[{0}] {1}" -f (Get-Date -Format 's'), $m) }
try {
    $cur = Join-Path $dir 'current.inf'
    & secedit /export /cfg $cur /areas USER_RIGHTS | Out-Null
    if (-not (Test-Path (Join-Path $dir 'before.inf'))) { Copy-Item $cur (Join-Path $dir 'before.inf') }
    $lines = [System.Collections.Generic.List[string]]::new([string[]](Get-Content $cur))
    $entry = "*$Sid"
    $i = $lines.FindIndex([Predicate[string]] { param($l) $l -match '^SeLockMemoryPrivilege\s*=' })
    if ($Revoke) {
        if ($i -lt 0) { Log 'revoke: nothing granted'; exit 0 }
        $rest = ($lines[$i] -replace '^SeLockMemoryPrivilege\s*=\s*', '').Split(',') | ForEach-Object { $_.Trim() } |
            Where-Object { $_ -and $_ -ne $entry }
        $lines[$i] = 'SeLockMemoryPrivilege = ' + ($rest -join ',')
    } elseif ($i -ge 0) {
        if ($lines[$i] -match [regex]::Escape($entry)) { Log "already granted: $($lines[$i])"; exit 0 }
        $lines[$i] = $lines[$i].TrimEnd() + ",$entry"
    } else {
        $j = $lines.FindIndex([Predicate[string]] { param($l) $l -match '^\[Privilege Rights\]' })
        if ($j -lt 0) { throw 'no [Privilege Rights] section in the secedit export' }
        $lines.Insert($j + 1, "SeLockMemoryPrivilege = $entry")
    }
    $new = Join-Path $dir 'new.inf'
    Set-Content -Path $new -Value $lines -Encoding Unicode
    & secedit /configure /db (Join-Path $dir 'new.sdb') /cfg $new /areas USER_RIGHTS /quiet | Out-Null
    Log "secedit /configure exit $LASTEXITCODE ($(if ($Revoke) { 'revoke' } else { 'grant' }) $Sid)"
    if ($LASTEXITCODE -ne 0) { exit 1 }
} catch {
    Log "ERROR: $($_.Exception.Message)"
    exit 1
}
