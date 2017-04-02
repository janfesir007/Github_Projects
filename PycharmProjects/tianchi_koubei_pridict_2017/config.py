# -*-coding:gbk-*-
# -*-coding:utf-8-*-
import datetime
names_file = 'data/process_data/user_pay_del_zero.csv'
start = datetime.datetime(2015, 07, 01)
end = datetime.datetime(2016, 10, 31)
save_file = 'data/process_data/user_pay_{}_{}.npz'.format(datetime.datetime.strftime(start, "%Y%m%d"),
        datetime.datetime.strftime(end, "%Y%m%d"))
save_file_diff = 'data/process_data/diff_user_pay_{}_{}.npz'.format(datetime.datetime.strftime(start, "%Y%m%d"),
        datetime.datetime.strftime(end, "%Y%m%d"))
save_file_diff_normalize = 'data/process_data/diff_normalize_user_pay_{}_{}.npz'.format(datetime.datetime.strftime(start, "%Y%m%d"),
        datetime.datetime.strftime(end, "%Y%m%d"))
normalize_std_len = 50

class SmallConfig(object):
    """Small config."""
    init_scale = 0.1  # ��ز����ĳ�ʼֵΪ������ȷֲ�����Χ��[-init_scale,+init_scale]
    learning_rate = 1.0  # ѧϰ����,ֵԽ��,����شﵽ����ֵ,��Ҫ��ֹ��ͷ.��ѭ����������max_epoch�Ժ���𽥽���
    lr_decay = 0.5  # ѧϰ��˥��
    max_grad_norm = 5  # ���ڿ����ݶ�����,����ݶ�������L2ģ����max_grad_norm����ȱ�����С
    max_epoch = 5  # epoch<max_epochʱ,lr_decayֵ=1,epoch>max_epochʱ,lr_decay�𽥼�С
    max_max_epoch = 10  # ������ѭ������
    num_layers = 2  # LSTM����,������(ÿ�����һ��ѭ��������,����һ��ʱ������)
    num_steps = 10  # the number of unrolled steps of LSTM;ʱ���ĸ���(ʱ�����е�����)
    batch_size = 30  # ���ݿ���
    hidden_size = 100  # the number of LSTM units
    keep_prob = 1.0  # ����dropout.ÿ����������ʱ�������е�ÿ����Ԫ����1-keep_prob�ĸ��ʲ�����,���Է�ֹ�����


class MediumConfig(object):
    """Medium config."""
    init_scale = 0.05
    learning_rate = 0.1
    lr_decay = 0.8
    max_grad_norm = 5
    max_epoch = 1 # epoch<max_epochʱ,lr_decayֵ=1,epoch>max_epochʱ,lr_decay�𽥼�С
    max_max_epoch = 4
    num_layers = 2
    num_steps = 12
    batch_size = 30
    hidden_size = 200
    keep_prob = 0.8


class LargeConfig(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 0.01
    lr_decay = 1 / 1.15
    max_grad_norm = 10
    max_epoch = 5  # epoch<max_epochʱ,lr_decayֵ=1,epoch>max_epochʱ,lr_decay�𽥼�С
    max_max_epoch = 10
    num_layers = 2
    num_steps = 10
    batch_size = 30
    hidden_size = 1500
    keep_prob = 0.35
