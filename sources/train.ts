import { Model, tensor2d, tensor4d, Tensor2D, Tensor4D } from '@tensorflow/tfjs'
import '@tensorflow/tfjs-node'
import { Signale } from 'signale'
import { flatten } from 'lodash'

const ilog: Signale = new Signale({
  interactive: true
})

const log: Signale = new Signale()

const oneHotEncode = (numDimension: number, hotValue: number): number[] =>
  Array.from(new Array(numDimension)).map((_: any, i: number) =>
    i === hotValue ? 1 : 0
  )

export const trainModel = async (
  model: Model,
  trainBatchSize: number,
  trainData: any[],
  trainLabels: number[],
  testBatch: number
): Promise<void> => {
  // Compiling the model with rmsprop optimizer
  model.compile({
    optimizer: 'rmsprop',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  })

  for (
    let idx = 0;
    idx + trainBatchSize < trainData.length;
    idx += trainBatchSize
  ) {
    ilog.info('Starting new batch')
    const data = tensor4d(flatten(trainData.slice(idx, idx + trainBatchSize)), [
      trainBatchSize,
      28,
      28,
      1
    ])
    const labels = tensor2d(
      flatten(trainLabels.slice(idx, idx + trainBatchSize)).map((val: number) =>
        oneHotEncode(10, val)
      ),
      [trainBatchSize, 10]
    )

    let validation: [Tensor4D, Tensor2D]

    if ((idx / trainBatchSize) % 5 === 0) {
      let valData: any = []
      const valLabels: any[] = []
      for (let idx = 0; idx < testBatch; ++idx) {
        const randSelection: number = Math.floor(
          Math.random() * trainData.length
        )
        valData = valData.concat(trainData[randSelection])
        valLabels.push(trainLabels[randSelection])
      }
      validation = [
        tensor4d(valData, [testBatch, 28, 28, 1]),
        tensor2d(
          valLabels.map((val: number): number[] => oneHotEncode(10, val)),
          [testBatch, 10]
        )
      ]
    }

    ilog.info('Fitting')
    const history = await model.fit(data, labels, {
      batchSize: trainBatchSize,
      validationData: validation,
      epochs: 3,
      verbose: 0
    })
    ilog.success(
      `Fitting round complete with batch of size ${trainBatchSize} (${idx} / ${
        trainData.length
      }) | accuracy is ${history.history.acc[0]}`
    )
    process.stdout.write('\n')
  }

  // Running final model evaluation for output purposes
  let valData: any = []
  const valLabels: any[] = []
  for (let randIdx = 0; randIdx < testBatch; ++randIdx) {
    const randSelection: number = Math.floor(Math.random() * trainData.length)
    valData = valData.concat(trainData[randSelection])
    valLabels.push(trainLabels[randSelection])
  }
  const validation = [
    tensor4d(valData, [testBatch, 28, 28, 1]),
    tensor2d(valLabels.map((val: number): number[] => oneHotEncode(10, val)), [
      testBatch,
      10
    ])
  ]
  const result = model.evaluate(validation[0], validation[1], {
    batchSize: testBatch
  })
  log.info('Loss', (result as any)[0].dataSync()[0])
  log.info('Accuracy', (result as any)[1].dataSync()[0])
}
