import utils from './utils.js'
import path from 'path'

const COMPONENT_MAP_LOCAL = {
  agent: { src: 'agent', dest: 'agent' },
  commands: { src: 'commands', dest: 'commands' },
  skills: { src: 'skills', dest: 'skill' },
}

const COMPONENT_MAP_GLOBAL = {
  core: { src: '', dest: '' },
}

async function installLocal(projectRoot, components, options = {}) {
  const { force = false } = options
  const corePath = utils.getCorePath(projectRoot)
  const localPath = utils.getLocalInstallPath(projectRoot)

  utils.log.info(`本项目安装模式`)
  utils.log.info(`源路径: ${corePath}`)
  utils.log.info(`目标路径: ${localPath}`)

  const results = []

  for (const comp of components) {
    const mapping = COMPONENT_MAP_LOCAL[comp]
    if (!mapping) {
      utils.log.error(`未知组件: ${comp}`)
      continue
    }

    const srcPath = path.join(corePath, mapping.src)
    const destPath = path.join(localPath, mapping.dest)

    utils.log.info(`安装组件: ${comp}`)

    const exclude = comp === 'skills' ? [] : []
    const copied = await utils.copyDir(srcPath, destPath, { force, exclude })

    results.push({ component: comp, copied, path: destPath })
  }

  await utils.recordInstall(projectRoot, components, 'local')

  return results
}

async function installGlobal(projectRoot, components, options = {}) {
  const { force = false } = options
  const corePath = utils.getCorePath(projectRoot)
  const globalPath = utils.getGlobalInstallPath()

  utils.log.info(`全局安装模式`)
  utils.log.info(`源路径: ${corePath}`)
  utils.log.info(`目标路径: ${globalPath}`)

  await utils.copyDir(corePath, globalPath, { force, exclude: ['install'] })

  await utils.recordInstall(projectRoot, ['all'], 'global')

  return [{ component: 'all', copied: true, path: globalPath }]
}

export default {
  installLocal,
  installGlobal,
  COMPONENT_MAP_LOCAL,
  COMPONENT_MAP_GLOBAL,
}
