import { Model, tensor4d, Tensor2D } from '@tensorflow/tfjs'
import * as Signale from 'signale'
import '@tensorflow/tfjs-node'

const printAsciiArt = (testSet: any[], randDraw: number): void => {
  console.log('-'.repeat(56))
  for (let print_idx_y = 0; print_idx_y < 28; ++print_idx_y) {
    for (let print_idx_x = 0; print_idx_x < 28; ++print_idx_x) {
      const val = testSet[randDraw][print_idx_y * 28 + print_idx_x]
      let char = ' '
      if (val > 200) {
        char = '8'
      } else if (val > 150) {
        char = 'o'
      } else if (val > 100) {
        char = '.'
      }

      process.stdout.write(char.repeat(2))
    }
    process.stdout.write('\n')
  }
}

export const predict = async (
  model: Model,
  testSet: any[],
  count: number
): Promise<void> => {
  for (let idx = 0; idx < count; ++idx) {
    const randDraw: number = Math.floor(Math.random() * testSet.length)
    const result: Tensor2D = model.predict(
      tensor4d(testSet[randDraw], [1, 28, 28, 1])
    ) as Tensor2D
    printAsciiArt(testSet, randDraw)
    Signale.info(
      `It's a ${result.dataSync().indexOf(Math.max(...result.dataSync()))} !`
    )
    process.stdout.write('\n')
  }
}
