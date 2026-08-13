#!/usr/bin/env node
import { program } from 'commander'
import installer from './installer.js'
import utils from './utils.js'
import path from 'path'

program
  .name('opencode-installer')
  .description('Install core capabilities to local .opencode/ or global path')
  .version('1.0.0')
  .option('--global', 'Install globally to ~/.config/opencode/', false)
  .option('--components <list>', 'Components to install (agent,commands,skills,all)', 'all')
  .option('--force', 'Overwrite existing files', false)
  .action(async (options) => {
    const projectRoot = process.cwd()

    let components
    if (options.components === 'all') {
      components = ['agent', 'commands', 'skills']
    } else {
      components = options.components.split(',').map((c) => c.trim())
    }

    const installOptions = { force: options.force }

    let results
    if (options.global) {
      results = await installer.installGlobal(projectRoot, components, installOptions)
    } else {
      results = await installer.installLocal(projectRoot, components, installOptions)
    }

    utils.log.success('安装完成！')
    results.forEach((r) => {
      const status = r.copied ? '✓' : '⊘'
      console.log(`  ${status} ${r.component} → ${r.path}`)
    })
  })

program.parse()
