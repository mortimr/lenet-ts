import { Model } from '@tensorflow/tfjs';
import '@tensorflow/tfjs-node';

export async function save_model(model: Model, path: string): Promise<void> {
    await model.save(path);
}
