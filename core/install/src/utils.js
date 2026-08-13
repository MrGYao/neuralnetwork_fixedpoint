import fs from 'fs-extra'
import path from 'path'
import os from 'os'

const log = {
  info: (msg) => console.log(`\x1b[36mℹ\x1b[0m ${msg}`),
  success: (msg) => console.log(`\x1b[32m✓\x1b[0m ${msg}`),
  warn: (msg) => console.log(`\x1b[33m⚠\x1b[0m ${msg}`),
  error: (msg) => console.log(`\x1b[31m✗\x1b[0m ${msg}`),
}

function getGlobalInstallPath() {
  return path.join(os.homedir(), '.config', 'opencode')
}

function getLocalInstallPath(projectRoot) {
  return path.join(projectRoot, '.opencode')
}

function getCorePath(projectRoot) {
  return path.join(projectRoot, 'core')
}

async function copyDir(src, dest, options = {}) {
  const { force = false, exclude = [] } = options

  if (await fs.pathExists(dest)) {
    if (!force) {
      log.warn(`目标已存在，跳过: ${dest}`)
      return false
    }
    log.info(`覆盖目录: ${dest}`)
    await fs.remove(dest)
  }

  await fs.ensureDir(dest)

  const items = await fs.readdir(src)
  for (const item of items) {
    if (exclude.includes(item)) continue

    const srcPath = path.join(src, item)
    const destPath = path.join(dest, item)
    const stat = await fs.stat(srcPath)

    if (stat.isDirectory()) {
      await copyDir(srcPath, destPath, options)
    } else {
      await fs.copy(srcPath, destPath)
    }
  }

  return true
}

async function getInstallHistoryPath(projectRoot) {
  return path.join(projectRoot, 'core', 'install-history.json')
}

async function recordInstall(projectRoot, components, mode) {
  const historyPath = await getInstallHistoryPath(projectRoot)
  const history = (await fs.pathExists(historyPath))
    ? await fs.readJson(historyPath)
    : { installs: [] }

  history.installs.push({
    time: new Date().toISOString(),
    mode,
    components,
  })

  await fs.writeJson(historyPath, history, { spaces: 2 })
  return historyPath
}

export default {
  log,
  getGlobalInstallPath,
  getLocalInstallPath,
  getCorePath,
  copyDir,
  recordInstall,
}
