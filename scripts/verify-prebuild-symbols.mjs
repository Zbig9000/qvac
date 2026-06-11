#!/usr/bin/env node
// scripts/verify-prebuild-symbols.mjs
//
// Static guard against the "addon loads fine on its own but dlopen-crashes once
// it is co-loaded with other ggml addons in the SDK" class of bug.
//
// Concretely this is the @qvac/tts-ggml@0.2.1 regression: the addon `.bare`
// shipped with UND `ggml_backend_is_cpu` / `ggml_get_type_traits_cpu` in its
// dynamic symbol table and NO `DT_NEEDED` providing them (those symbols live in
// the lazily-dlopen'd micro-arch CPU backend `.so`s). Engine symbols are never
// provided by the Bare host process, so an unresolved one is a guaranteed
// dlopen/SIGABRT on device. That is deterministically detectable here, with no
// emulator or device, straight off the prebuild artifacts.
//
// For every prebuilt addon binary this performs two checks:
//
//   1. UND check (HARD FAIL): the addon module must not leave any ENGINE symbol
//      (ggml_/gguf_/llama_/whisper_/... by default) UNDEFINED unless one of its
//      own declared DT_NEEDED libraries (that ships alongside it) defines it.
//      Lazily dlopen'd backend libs do NOT count -- they are not loaded when the
//      module itself is dlopen'd, which is exactly why 0.2.1 crashed.
//
//   2. Export-hygiene check: the addon module should export only `bare_*` /
//      `napi_*` (plus the host C/C++ ABI), never `ggml_*` / `llama_*` / ...
//      Two co-loaded addons that each export their own ggml copy can interpose
//      on one another and corrupt the backend registry. Default: report only.
//      Pass `--enforce-exports` to turn leaked engine exports into a hard fail
//      (see packages/transcription-parakeet/symbols.map for the fix template).
//
// Usage:
//   node scripts/verify-prebuild-symbols.mjs --dir <prebuilds-dir> [options]
//   node scripts/verify-prebuild-symbols.mjs <prebuilds-dir>
//
// Options:
//   --dir <path>            Directory to scan recursively (repeatable). Default: ./prebuilds
//   --platform <p>          linux|android|darwin|ios|win32 (informational; affects messaging)
//   --engine-prefixes <csv> Symbol roots treated as engine-internal.
//                           Default: ggml,gguf,llama,whisper,clip,mtmd,sd,stable_diffusion
//   --enforce-exports       Make leaked engine exports a hard failure (default: warn)
//   --allow-export <name>   Allowlist a specific exported symbol (repeatable)
//   --readelf <path>        readelf binary (default: $READELF or llvm-readelf or readelf)
//   --nm <path>             nm binary (default: $NM or llvm-nm or nm)
//   --json                  Emit a machine-readable JSON report to stdout
//   --quiet                 Only print failures
//
// Exit code: 0 = clean, 1 = at least one hard failure, 2 = bad usage / tooling.

import { readdirSync, openSync, readSync, closeSync } from 'node:fs'
import { join, basename, resolve } from 'node:path'
import { execFileSync } from 'node:child_process'

const IS_GHA = process.env.GITHUB_ACTIONS === 'true'

function ghaError (msg) {
  console.error(IS_GHA ? `::error::${msg}` : `ERROR: ${msg}`)
}
function ghaWarn (msg) {
  console.error(IS_GHA ? `::warning::${msg}` : `WARNING: ${msg}`)
}

function fail (msg) {
  ghaError(msg)
  process.exit(2)
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

function parseArgs (argv) {
  const opts = {
    dirs: [],
    platform: process.env.PREBUILD_PLATFORM || '',
    enginePrefixes: ['ggml', 'gguf', 'llama', 'whisper', 'clip', 'mtmd'],
    enforceExports: false,
    allowExports: new Set(),
    readelf: process.env.READELF || '',
    nm: process.env.NM || '',
    json: false,
    quiet: false,
  }
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]
    switch (a) {
      case '--dir': opts.dirs.push(argv[++i]); break
      case '--platform': opts.platform = argv[++i]; break
      case '--engine-prefixes': opts.enginePrefixes = argv[++i].split(',').map(s => s.trim()).filter(Boolean); break
      case '--enforce-exports': opts.enforceExports = true; break
      case '--allow-export': opts.allowExports.add(argv[++i]); break
      case '--readelf': opts.readelf = argv[++i]; break
      case '--nm': opts.nm = argv[++i]; break
      case '--json': opts.json = true; break
      case '--quiet': opts.quiet = true; break
      case '-h': case '--help': printHelp(); process.exit(0); break
      default:
        if (a.startsWith('--')) fail(`Unknown option: ${a}`)
        else opts.dirs.push(a)
    }
  }
  if (opts.dirs.length === 0) opts.dirs.push('prebuilds')
  return opts
}

function printHelp () {
  const text = process.argv[1] ? `node ${basename(process.argv[1])}` : 'verify-prebuild-symbols'
  console.log(`Usage: ${text} --dir <prebuilds-dir> [--platform p] [--enforce-exports] [--engine-prefixes csv] [--json]`)
}

// ---------------------------------------------------------------------------
// Tool resolution
// ---------------------------------------------------------------------------

function which (candidates) {
  for (const c of candidates) {
    if (!c) continue
    try {
      execFileSync(c, ['--version'], { stdio: 'ignore' })
      return c
    } catch { /* try next */ }
  }
  return ''
}

function resolveTools (opts) {
  const readelf = which([opts.readelf, 'llvm-readelf', 'readelf'])
  const nm = which([opts.nm, 'llvm-nm', 'nm'])
  if (!nm) fail('No usable nm (tried llvm-nm, nm). Install LLVM or binutils.')
  // readelf is only needed for ELF DT_NEEDED resolution; Mach-O does not use it.
  return { readelf, nm }
}

// ---------------------------------------------------------------------------
// Binary discovery + format detection
// ---------------------------------------------------------------------------

const BIN_EXT = /\.(bare|so|dylib)(\.\d+)*$/i

function walk (dir, out) {
  let entries
  try { entries = readdirSync(dir, { withFileTypes: true }) } catch { return }
  for (const e of entries) {
    const p = join(dir, e.name)
    if (e.isDirectory()) walk(p, out)
    else if (e.isFile() && BIN_EXT.test(e.name)) out.push(p)
  }
}

function detectFormat (path) {
  let fd
  try {
    fd = openSync(path, 'r')
    const buf = Buffer.alloc(4)
    readSync(fd, buf, 0, 4, 0)
    if (buf[0] === 0x7f && buf[1] === 0x45 && buf[2] === 0x4c && buf[3] === 0x46) return 'elf'
    const m = buf.readUInt32BE(0)
    // Mach-O (32/64, BE/LE) + fat
    if ([0xfeedface, 0xfeedfacf, 0xcefaedfe, 0xcffaedfe, 0xcafebabe, 0xbebafeca].includes(m)) return 'macho'
    return 'unknown'
  } catch {
    return 'unknown'
  } finally {
    if (fd !== undefined) closeSync(fd)
  }
}

// `qvac__<pkg>.bare` (or libqvac__<pkg>.<ver>.so) is the module Bare dlopen's.
// `lib...ggml-<backend>...so` are the lazily-loaded backend modules; they are
// SUPPOSED to have UND ggml-base symbols (resolved against the module at load)
// and to export ggml symbols, so they are exempt from the strict checks.
function classify (path) {
  const b = basename(path)
  if (/qvac__/.test(b) || b.endsWith('.bare')) return 'module'
  if (/lib.*ggml[-_.]/i.test(b)) return 'backend'
  return 'other'
}

// ---------------------------------------------------------------------------
// Symbol extraction (POSIX `nm -P`: "name type value size")
// ---------------------------------------------------------------------------

function runNm (nm, args, path) {
  try {
    return execFileSync(nm, [...args, path], { encoding: 'utf8', maxBuffer: 256 * 1024 * 1024 })
  } catch (e) {
    // nm exits non-zero for "no symbols"; still returns partial stdout.
    if (e.stdout) return e.stdout.toString()
    return ''
  }
}

function parseNmPosix (output) {
  const undefined_ = new Set()
  const exported = new Set()
  for (const line of output.split('\n')) {
    if (!line) continue
    const sp = line.indexOf(' ')
    if (sp <= 0) continue
    const rawName = line.slice(0, sp)
    // Normalize ELF symbol versioning (`foo@GLIBC_2.2.5`, `foo@@ZLIB_1.2.0`) to
    // the base name so a versioned export satisfies an (un)versioned import.
    const name = rawName.split('@')[0]
    const rest = line.slice(sp + 1).trimStart()
    const type = rest[0]
    if (!type) continue
    if (type === 'U') undefined_.add(name) // strong undefined (weak 'v'/'w' won't break load)
    else if (type >= 'A' && type <= 'Z') exported.add(name) // defined, external (incl. weak W/V, absolute A)
  }
  return { undefined_, exported }
}

function symbolsOf (tools, path, format) {
  // ELF uses the dynamic symbol table (-D); Mach-O has a single table.
  const args = format === 'elf' ? ['-D', '-P'] : ['-P']
  return parseNmPosix(runNm(tools.nm, args, path))
}

function neededOf (tools, path, format) {
  if (format !== 'elf' || !tools.readelf) return []
  let out = ''
  try {
    out = execFileSync(tools.readelf, ['-d', '--wide', path], { encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 })
  } catch (e) { out = e.stdout ? e.stdout.toString() : '' }
  const needed = []
  for (const line of out.split('\n')) {
    const m = line.match(/\(NEEDED\)\s+Shared library:\s+\[([^\]]+)\]/)
    if (m) needed.push(m[1])
  }
  return needed
}

// ---------------------------------------------------------------------------
// Core check
// ---------------------------------------------------------------------------

function isEngineSymbol (name, prefixes) {
  // Word-boundary match only: `ggml_backend_is_cpu` matches `ggml`, but
  // `sdl_init` must NOT match `sd`. Engine C-ABI symbols are always `<root>_*`
  // (or the bare root), so this avoids false positives on unrelated symbols.
  for (const p of prefixes) {
    if (name === p || name.startsWith(p + '_')) return true
  }
  return false
}

function main () {
  const opts = parseArgs(process.argv.slice(2))
  const tools = resolveTools(opts)

  const binaries = []
  for (const d of opts.dirs) walk(resolve(d), binaries)

  if (binaries.length === 0) {
    // skip-prebuilds addons (e.g. decoder-audio) legitimately ship nothing here.
    if (!opts.quiet) console.log(`verify-prebuild-symbols: no .bare/.so/.dylib found under ${opts.dirs.join(', ')} -- nothing to check.`)
    if (opts.json) console.log(JSON.stringify({ ok: true, checked: 0, results: [] }))
    process.exit(0)
  }

  // Pre-index exported symbols of every binary by soname so DT_NEEDED lookups
  // can resolve against co-located libraries.
  const exportsBySoname = new Map()
  const fmtByPath = new Map()
  for (const p of binaries) {
    const fmt = detectFormat(p)
    fmtByPath.set(p, fmt)
    if (fmt === 'unknown') continue
    const { exported } = symbolsOf(tools, p, fmt)
    exportsBySoname.set(basename(p), exported)
  }

  const results = []
  let hardFail = false

  for (const p of binaries) {
    const fmt = fmtByPath.get(p)
    const kind = classify(p)
    const rel = p
    if (fmt === 'unknown') { ghaWarn(`Could not detect binary format: ${rel}`); continue }
    if (kind === 'backend') continue // backend libs legitimately import/export engine symbols

    const { undefined_, exported } = symbolsOf(tools, p, fmt)

    // Provider set = exported symbols of co-located DT_NEEDED libs.
    const needed = neededOf(tools, p, fmt)
    const provided = new Set()
    for (const soname of needed) {
      const exp = exportsBySoname.get(soname)
      if (exp) for (const s of exp) provided.add(s)
    }

    const unresolvedEngine = []
    for (const s of undefined_) {
      if (!isEngineSymbol(s, opts.enginePrefixes)) continue
      if (provided.has(s)) continue
      unresolvedEngine.push(s)
    }

    const leakedExports = []
    for (const s of exported) {
      if (!isEngineSymbol(s, opts.enginePrefixes)) continue
      if (opts.allowExports.has(s)) continue
      if (/^(bare_|napi_)/.test(s)) continue
      leakedExports.push(s)
    }

    const fileFail = unresolvedEngine.length > 0 || (opts.enforceExports && leakedExports.length > 0)
    if (fileFail) hardFail = true

    results.push({ path: rel, kind, format: fmt, unresolvedEngine, leakedExports })

    if (unresolvedEngine.length > 0) {
      ghaError(`${rel}: ${unresolvedEngine.length} unresolved engine symbol(s) (will dlopen-crash on device): ${unresolvedEngine.sort().slice(0, 20).join(', ')}${unresolvedEngine.length > 20 ? ', ...' : ''}`)
    }
    if (leakedExports.length > 0) {
      const m = `${rel}: exports ${leakedExports.length} engine symbol(s) at default visibility (cross-addon interposition risk): ${leakedExports.sort().slice(0, 20).join(', ')}${leakedExports.length > 20 ? ', ...' : ''}. Add a symbols.map (see transcription-parakeet) / -Wl,--exclude-libs,ALL.`
      if (opts.enforceExports) ghaError(m); else ghaWarn(m)
    }
    if (!opts.quiet && unresolvedEngine.length === 0 && leakedExports.length === 0) {
      console.log(`OK  ${rel} (${kind})`)
    }
  }

  if (opts.json) console.log(JSON.stringify({ ok: !hardFail, checked: results.length, results }, null, 2))

  if (hardFail) {
    ghaError('verify-prebuild-symbols: FAILED. See messages above.')
    process.exit(1)
  }
  if (!opts.quiet) console.log(`verify-prebuild-symbols: OK (${results.length} module/other binaries checked).`)
  process.exit(0)
}

main()
