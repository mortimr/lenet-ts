import { load }         from './load';
import { Signale }      from 'signale';
import { build_model }  from './model';
import { train_model }  from './train';
import { save_model }   from './save';
import { import_model } from './import';
import { predict }      from './predict';

const store: any = {};

const ilog: Signale = new Signale({
    interactive: true
});

const log: Signale = new Signale();

const train = async (): Promise<void> => {
    try {

        ilog.info('Loading Training Set');
        store['train'] = load(process.argv[3], true);
        ilog.success('Loaded Training Set');
        process.stdout.write('\n');

        const model = build_model();
        await train_model(model,
            42, store['train'].map((e: any) => e.data), store['train'].map((e: any) => e.label), 50);
        await save_model(model, 'file://' + process.argv[4]);
    } catch (e) {
        log.fatal(e);
        process.exit(1);
    }
};

const run = async (): Promise<void> => {
    try {
        ilog.info('Loading Test Set');
        store['test'] = load(process.argv[3], false);
        ilog.success('Loaded Test Set');
        process.stdout.write('\n');

        const model = await import_model(process.argv[4]);
        await predict(model, store['test'], parseInt(process.argv[5]));

    } catch (e) {
        log.fatal(e);
        process.exit(1);
    }
};

switch (process.argv[2]) {
    case 'train':
        train();
        break;
    case 'run':
        run();
        break;
    default:
        log.fatal('Use train to train and save the model. Use run to predict the test set.');
        process.exit(1);
}
