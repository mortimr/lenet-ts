import { Model, tensor2d, tensor4d, Tensor2D, Tensor4D } from '@tensorflow/tfjs';
import '@tensorflow/tfjs-node';
import { Signale }                                       from 'signale';

const ilog: Signale = new Signale({
    interactive: true
});

const log: Signale = new Signale();

export async function train_model(model: Model, train_batch: number, train_data: any[], train_labels: number[], test_batch: number): Promise<void> {
    // Compiling the model with rmsprop optimizer
    model.compile({
        optimizer: 'rmsprop',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    for (let idx = 0; idx < train_data.length && idx + train_batch < train_data.length; idx += train_batch) {

        ilog.info('Starting new batch');
        const data = tensor4d([].concat.apply([], train_data.slice(idx, idx + train_batch)), [train_batch, 28, 28, 1]);
        const labels = tensor2d([].concat.apply([], train_labels.slice(idx, idx + train_batch)
            .map((val: number): number[] => {
                // Manual and ugly one-hot conversion
                const ret: number[] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
                ret[val] = 1;
                return ret;
            })), [train_batch, 10]);

        let validation: [Tensor4D, Tensor2D];

        if ((idx / train_batch) % 5 === 0) {
            let val_data: any = [];
            const val_labels: any[] = [];
            for (let rand_idx = 0; rand_idx < test_batch; ++rand_idx) {
                const rand_selection: number = Math.floor(Math.random() * train_data.length);
                val_data = val_data.concat(train_data[rand_selection]);
                val_labels.push(train_labels[rand_selection]);
            }
            validation = [
                tensor4d([].concat.apply([], val_data), [test_batch, 28, 28, 1]),
                tensor2d([].concat.apply([], val_labels
                    .map((val: number): number[] => {
                        // Manual and ugly one-hot conversion
                        const ret: number[] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
                        ret[val] = 1;
                        return ret;
                    })), [test_batch, 10])
            ];
        }

        ilog.info('Fitting');
        const history = await model.fit(
            data,
            labels,
            {
                batchSize: train_batch,
                validationData: validation,
                epochs: 3,
                verbose: 0
            }
        );
        ilog.success(`Fitting round complete with batch of size ${train_batch} (${idx} / ${train_data.length}) | accuracy is ${history.history.acc[0]}`);
        process.stdout.write('\n');
    }

    // Running final model evaluation for output purposes
    let validation: [Tensor4D, Tensor2D];

    let val_data: any = [];
    const val_labels: any[] = [];
    for (let rand_idx = 0; rand_idx < test_batch; ++rand_idx) {
        const rand_selection: number = Math.floor(Math.random() * train_data.length);
        val_data = val_data.concat(train_data[rand_selection]);
        val_labels.push(train_labels[rand_selection]);
    }
    validation = [
        tensor4d([].concat.apply([], val_data), [test_batch, 28, 28, 1]),
        tensor2d([].concat.apply([], val_labels
            .map((val: number): number[] => {
                const ret: number[] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
                ret[val] = 1;
                return ret;
            })), [test_batch, 10])
    ];
    const result = model.evaluate(validation[0], validation[1], {batchSize: test_batch});
    log.info('Loss', (<any> result)[0].dataSync()[0]);
    log.info('Accuracy', (<any> result)[1].dataSync()[0]);

}
