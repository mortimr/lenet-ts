import { Model } from '@tensorflow/tfjs';
import '@tensorflow/tfjs-node';
import { SaveResult } from '@tensorflow/tfjs-core/dist/io/io';

export const saveModel = async (
  model: Model,
  path: string
): Promise<SaveResult> => model.save(path);
