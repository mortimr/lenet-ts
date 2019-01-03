import { Model, loadModel } from '@tensorflow/tfjs';
import '@tensorflow/tfjs-node';

export async function import_model(path: string): Promise<Model> {
    return loadModel('file://' + path);
}
