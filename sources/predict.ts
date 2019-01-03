import { Model, tensor4d, Tensor2D } from '@tensorflow/tfjs';
import * as Signale                  from 'signale';
import '@tensorflow/tfjs-node';

export async function predict(model: Model, test_set: any[], count: number): Promise<void> {
    for (let idx = 0; idx < count; ++idx) {
        const rand_draw: number = Math.floor(Math.random() * test_set.length);
        const result: Tensor2D = model.predict(tensor4d(test_set[rand_draw], [1, 28, 28, 1])) as Tensor2D;
        console.log('-'.repeat(56));
        for (let print_idx_y = 0; print_idx_y < 28; ++print_idx_y) {
            for (let print_idx_x = 0; print_idx_x < 28; ++print_idx_x) {
                const val = test_set[rand_draw][print_idx_y * 28 + print_idx_x];
                let char = ' ';
                if (val > 200) {
                    char = '8';
                } else if (val > 150) {
                    char = 'o';
                } else if (val > 100) {
                    char = '.';
                }

                process.stdout.write(char.repeat(2));
            }
            process.stdout.write('\n');
        }
        Signale.info(`It's a ${result.dataSync().indexOf(Math.max(...result.dataSync()))} !`);
        process.stdout.write('\n');
    }
}
