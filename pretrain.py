import paddle
from paddlenlp.transformers import UnifiedTransformerTokenizer
from model import UnifiedStateTransformerLMHeadModel
from utils import PretrainDataset
from tqdm import tqdm

model_name = 'plato-mini'
load_from = 'plato-mini'
save_to = 'chinese_sformer_L-6_H-768_A-12_S-8'
data_root = 'corpus'
batch_size = 4  # aistudio可使用16
max_data_epoch = 16
steps = 100000
num_workers = 4
state_size = 8
init_lr = 5e-5

tokenizer = UnifiedTransformerTokenizer.from_pretrained(model_name)
model = UnifiedStateTransformerLMHeadModel.from_pretrained(load_from)
dataset = PretrainDataset(tokenizer, data_root, batch_size=batch_size, steps=steps, max_data_epoch=max_data_epoch)
dataloader = paddle.io.DataLoader(dataset, return_list=True, batch_size=None, num_workers=num_workers)
loss_fn = paddle.nn.loss.CrossEntropyLoss(reduction='sum')
try:  # 随机初始化状态向量
    init_state = paddle.load(load_from + '/init_state.pdtensor')
    assert paddle.shape(init_state)[1] == state_size
except:
    print('\nInit state not found!\n')
    init_state = paddle.create_parameter(shape=[1, state_size, 768], dtype=paddle.float32,
                                         default_initializer=paddle.nn.initializer.TruncatedNormal(std=0.02))
opt = paddle.optimizer.Adam(init_lr, parameters=model.parameters() + [init_state])


def compute(data):
    correct, total, loss = 0, 0, 0
    initial_inputs = data['initial_inputs']
    masked_inputs = data['masked_inputs']
    labels = data['labels']
    bs = labels[0].shape[0]  # batch_size
    state = paddle.expand(init_state, (bs, state_size, 768))
    for i, model_input in enumerate(masked_inputs):
        model_input['state'] = state
        output, _ = model(**model_input)
        mask = paddle.cast(labels[i], dtype='bool')
        if paddle.any(mask):
            y_true = labels[i][mask]  # 只计算非0ids的loss, shape=(?,)
            y_pred = output[mask]  # shape=(?, vocab_size)
            loss += loss_fn(y_pred, y_true)
            y_pred = paddle.argmax(y_pred, axis=-1)
            total += y_pred.shape[0]
            correct += paddle.sum(paddle.cast(paddle.equal(y_pred, y_true), paddle.int32))
        if i < len(masked_inputs) - 1:  # 使用未mask的原始数据更新state
            model_input = initial_inputs[i]
            model_input['state'] = state
            _, state = model(**model_input)
    return {
        'loss': loss,
        'correct': correct,
        'total': total
    }


def train():
    total, correct, total_loss = 0, 0, 0
    model.train()
    for step, data in tqdm(enumerate(dataloader, start=1)):
        result = compute(data)
        total_loss += result['loss']
        total += result['total']
        correct += result['correct']
        result['loss'].backward()
        opt.step()
        opt.clear_grad()
        if step % 100 == 0:
            acc = correct / (total + 1e-9)
            loss = total_loss / (total + 1e-9)
            log = 'step: %d,\tlr: %f,\tloss: %f,\tacc: %f,\ttotal: %d\n' % (
                step, opt.get_lr(), loss, acc, total)
            print(log)
            total, correct, total_loss = 0, 0, 0  # 清零重新计算
            model.save_pretrained(save_to)
            paddle.save(init_state, save_to + '/init_state.pdtensor')
            with open(save_to + '/log.txt', 'a', encoding='utf-8') as f:
                f.write(log)


train()
