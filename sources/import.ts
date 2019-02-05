import { Model, loadModel } from '@tensorflow/tfjs'
import '@tensorflow/tfjs-node'

export const importModel = async (path: string): Promise<Model> =>
  loadModel('file://' + path)
