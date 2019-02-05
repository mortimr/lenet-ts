import { load } from './load'
import { Signale } from 'signale'
import { buildModel } from './model'
import { trainModel } from './train'
import { saveModel } from './save'
import { importModel } from './import'
import { predict } from './predict'

const store: any = {}

const ilog: Signale = new Signale({
  interactive: true
})

const log: Signale = new Signale()

const train = async (): Promise<void> => {
  ilog.info('Loading Training Set')
  store['train'] = load(process.argv[3], true)
  ilog.success('Loaded Training Set')
  process.stdout.write('\n')

  const model = buildModel()
  await trainModel(
    model,
    42,
    store['train'].map((e: any) => e.data),
    store['train'].map((e: any) => e.label),
    50
  )
  await saveModel(model, 'file://' + process.argv[4])
}

const run = async (): Promise<void> => {
  ilog.info('Loading Test Set')
  store['test'] = load(process.argv[3], false)
  ilog.success('Loaded Test Set')
  process.stdout.write('\n')

  const model = await importModel(process.argv[4])
  await predict(model, store['test'], parseInt(process.argv[5]))
}

const main = async (): Promise<void> => {
  switch (process.argv[2]) {
    case 'train':
      return train()
    case 'run':
      return run()
    default:
      throw new Error(
        'Use train to train and save the model. Use run to predict the test set.'
      )
  }
}

main().catch(
  (e: any): void => {
    log.fatal(e)
    process.exit(1)
  }
)
