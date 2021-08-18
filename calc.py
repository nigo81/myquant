#!/usr/bin python3
# -*- coding:UTF-8 -*-
# Author: nigo
import os
import json
import pandas as pd
import akshare as ak
import baostock as bs
import datetime
import numpy as np
from multiprocessing import Pool
import  multiprocessing.pool as mpp
import istarmap
from tqdm import tqdm
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from tabulate import tabulate
import wcwidth
import time
import re
import plotly.express as px
import plotly.graph_objects as go
import sys
import requests

PATH_CONFIG = './config.json'
PATH_TMP ='./tmp.csv'
with open(PATH_CONFIG,'rb') as f:
    json_str = json.load(f)
INDEX_LIST = json_str['index_list']
PATH_INFO = json_str['path_info']
PATH_INDEX = json_str['path_index']
PATH_WEIGHT = json_str['path_weight']
PATH_STOCK = json_str['path_stock']
PATH_MARKET = json_str['path_market']


def read_csv(path,**kw):
    df = pd.read_csv(path,converters=kw)
    return df

def full_code(code,is_index=True,is_dot=False):
    """补全证券代码
    code:6位证券代码
    is_index:是否是指数代码
    return:补全的代码
    """
    if is_dot:
        sh = 'sh.'
        sz = 'sz.'
    else:
        sh = 'sh'
        sz = 'sz'
    if is_index:
        if code[0] == '0':
            full = sh + code
        else:
            full = sz + code
    else:
        if code[0] == '6':
            full = sh + code
        else:
            full = sz + code
    return full


def convert_code(code):
    """指数代码小写转大写"""
    if code.endswith('sh'):
        return code[0:6] + '.SH'
    elif code.endswith('sz'):
        return code[0:6] + '.SZ'
    else:
        return code

def get_security_info(code=None):
    """获取指数基本信息
    code:6位代码
    return:代码对应信息
    """
    path = os.path.join(PATH_INFO,'index_info.csv')
    if os.path.exists(path):
        df = read_csv(path,index_code=str)
    else:
        print('获取指数基本信息列表')
        df = ak.index_stock_info()
        df.to_csv(path,index=False)
    if code:
        df = df[df.index_code==code]
    return df

def config_update_date(is_all=False):
    """获取配置文件中上次更新日期"""
    with open('config.json',mode='rb') as f:
        json_str = json.load(f)
        if is_all:
            update_date = json_str['update_date_all']
        else:
            update_date =  json_str['update_date']
        if update_date == "None":
            return None
        else:
            return update_date

def judge_update_stock(is_all=False):
    """判断是否更新数据"""
    update_date = config_update_date(is_all)
    if not update_date:
        return True
    if update_date < str(datetime.date.today()):
        return True
    else:
        return False

def judge_update_weight(date):
    """判断给定日期是否需要进行更新
    date:给定日期
    """
    update_date = config_update_date()
    if update_date:
        given_date = datetime.datetime.strptime(date,'%Y-%m-%d')
        update_date = datetime.datetime.strptime(update_date,'%Y-%m-%d')
        diff_month = (given_date.year - update_date.year) * 12 + given_date.month - update_date.month # 月份差
        if diff_month<1:
            return False
        else:
            return True
    else:
        return True

def get_exists_stocks_path():
    """获取本地文件已有数据的股票列表"""
    stock_files = os.listdir(PATH_STOCK)
    stocks = [ os.path.join(PATH_STOCK,file) for file in stock_files if file.endswith('csv')]
    return stocks

def get_all_stocks():
    """获取本地文件已有数据的股票列表"""
    stock_files = os.listdir(PATH_STOCK)
    stocks = [ file.split('_')[0] for file in stock_files if file.endswith('csv')]
    return stocks


def get_stocks(code,date):
    """计算给定日期的成分股"""
    df_new,df_history = get_index_weight(code,date)
    df_new = df_new[df_new.in_date <= date]
    df_history = df_history[(df_history.out_date > date) & (df_history.in_date <= date)]
    a = df_new['stock_code'].to_list()
    b = df_history['stock_code'].to_list()
    c = a + b
    stocks = list(set(c))
    return stocks

def get_all_index_stocks(index_list):
    """计算指数列表所涉及到的所有股票"""
    stock_list = []
    for code in index_list:
        df_new,df_history = get_index_weight(code)
        a = df_new['stock_code'].to_list()
        b = df_history['stock_code'].to_list()
        c = a + b
        stock_list += c
    stock_list = list(set(stock_list))
    if os.path.exists(PATH_TMP):
        os.remove(PATH_TMP)
    return stock_list

def get_index_weight(code,date=None):
    """获取指数成分股"""
    if not date:
        date = str(datetime.date.today())
    file_new = code + '-new.csv'
    path_new = os.path.join(PATH_WEIGHT,file_new)
    file_history = code + '-history.csv'
    path_history = os.path.join(PATH_WEIGHT,file_history)
    flag = judge_update_weight(date)
    # 判断是不是更新过了，应对接口频率限制
    if os.path.exists(PATH_TMP):
        tmp = read_csv(PATH_TMP,code=str)
    else:
        tmp = pd.DataFrame(columns=['code'])
    ignore_code_list = tmp['code'].to_list()
    if code in ignore_code_list:
        flag = False
    # 判断是不是更新过了，应对接口频率限制
    if os.path.exists(path_new) and not flag:
        df_new = read_csv(path_new,publish_date=str,stock_code=str)
    else:
        print('获取%s的成分股最新数据' % code)
        df_new = ak.index_stock_cons(code)
        df_new.columns = ['stock_code','stock_name','in_date']
        df_new.to_csv(path_new,index=False)
    if os.path.exists(path_history) and not flag:
        df_history = read_csv(path_history,publish_date=str,stock_code=str)
    else:
        print('获取%s的成分股历史数据' % code)
        df_history = ak.index_stock_hist(full_code(code))
        df_history.to_csv(path_history,index=False)
        df_updated = pd.DataFrame([code],columns=['code'])
        tmp = pd.concat([tmp,df_updated])
        tmp.to_csv(PATH_TMP,index=False)
    return df_new,df_history

def get_k_date(full_code,start_date,end_date):
    """获取单个股票数据"""
    rs = bs.query_history_k_data_plus(full_code,
        "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST",
        start_date=start_date, end_date=end_date, 
        frequency="d", adjustflag="3")
    result_list = []
    while (rs.error_code == '0') & rs.next():
        result_list.append(rs.get_row_data())
    df = pd.DataFrame(result_list, columns=rs.fields)
    return df

def get_trade_date(start_date=None,end_date=None):
    """获取交易日历
    start_date:开始日期
    end_date:结束日期
    """
    if not start_date:
        start_date = '1990-01-01'
    if not end_date:
        end_date = datetime.date.today()
        end_date = end_date.strftime('%Y-%m-%d')
    path = os.path.join(PATH_INFO,'a_trade_date.csv')
    flag = judge_update_stock() and judge_update_stock(is_all=True)
    if not flag and os.path.exists(path):
        print('读取交易日历')
        df = read_csv(path)
    else:
        print('获取交易日历')
        df = ak.tool_trade_date_hist_sina()
        df.columns = ['trade_date']
        df.to_csv(path,index=False)
        df['trade_date'] = df['trade_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    df = df[(df.trade_date<=end_date) & (df.trade_date>=start_date)]
    return df['trade_date'].to_list()

def get_hk_stock(code):
    """获取香港PE\PB"""
    df_pb = ak.stock_hk_eniu_indicator(symbol=code, indicator="市净率")
    df_pb = df_pb.loc[:,['date','pb']]
    df_pe = ak.stock_hk_eniu_indicator(symbol=code, indicator="市盈率")
    df_pe = df_pe.loc[:,['date','pe']]
    df = pd.merge(df_pe,df_pb,on='date')
    df['psTTM'] = ''
    df.columns = ['date','peTTM','pbMRQ','psTTM']
    path = os.path.join(PATH_STOCK,'%s_indicator.csv' % code)
    df.to_csv(path,index=False)

def update_stock_data(stocks,is_all=False,use_flag=False):
    """更新所有股票数据"""
    bs.login()
    today = datetime.date.today()
    today = str(today)
    yestoday = datetime.date.today() - datetime.timedelta(days=1)
    yestoday = str(yestoday)
    trade_date_list = get_trade_date(end_date=yestoday)
    end_date = trade_date_list[-1]
    pbar = tqdm(stocks)
    for stock in pbar:
        pbar.set_description("更新股票%s数据" % stock)
        # print('更新股票%s数据' % stock)
        path = os.path.join(PATH_STOCK,'%s_indicator.csv' % stock)
        if use_flag:
            flag = judge_update_stock(is_all)
        else:
            flag = True
        if os.path.exists(path):
            if flag:
                try:
                    df = read_csv(path,code=str,date=str)
                    data_date = df.iloc[-1,0]
                    if end_date<=data_date:
                        continue
                    else:
                        if stock[:2] == 'hk':
                            get_hk_stock(stock)
                        else:
                            start_date = datetime.datetime.strptime(data_date,'%Y-%m-%d') + datetime.timedelta(days=1)
                            start_date = start_date.strftime('%Y-%m-%d')
                            df_single = get_k_date(full_code(stock,is_index=False,is_dot=True),start_date,today)
                            if not df_single.empty:
                                df_single.to_csv(path,index=False,header=False,mode='a')
                except:
                    if stock[:2]  == 'hk':
                        get_hk_stock(stock)
                    else:
                        df_single = get_k_date(full_code(stock,is_index=False,is_dot=True),'1990-01-01',today)
                        if not df_single.empty:
                            df_single.to_csv(path,index=False)
        else:
            if stock[:2] == 'hk':
                get_hk_stock(stock)
            else:
                df_single = get_k_date(full_code(stock,is_index=False,is_dot=True),'1990-01-01',today)
                if not df_single.empty:
                    df_single.to_csv(path,index=False)
    bs.logout()

def calc_avg(numbers):
    """等权平均"""
    return len(numbers) / sum([1 / p if p > 0 else 0 for p in numbers])

def calc_mid(numbers):
    numbers = [ i for i in numbers if i>0]
    if numbers:
        numbers.sort()
        half = len(numbers) // 2
        return (numbers[half] + numbers[~half]) / 2
    else:
        return 0

def calc_average(*args,method='avg'):
    """根据pe_list,pb_list,ps_list取平均数
    args: (pe_list,pb_list,ps_list)
    method:计算模式 avg等权平均 mid中位数
    """
    num = len(args[0]) # 列表中元素数量
    result_list = []
    for arg in args:
        if method == 'mid':
            result = calc_mid(arg)
        else:
            result = calc_avg(arg)
        result = round(result,2)
        result_list.append(result)
    return tuple(result_list)

def get_index_pe_pb_date(date, stocks):
    '''指定日期的指数PE_PB（等权重）'''
    pe_list = []
    pb_list = []
    ps_list = []
    for stock in stocks:
        path = os.path.join(PATH_STOCK,'%s_indicator.csv' % stock)
        if not os.path.exists(path):
            print('股票%s文件不存在' % stock)
            continue
        df = read_csv(path,code=str,date=str)
        df_tmp = df[df.date == date]
        df_tmp = df_tmp.reset_index(drop=True) # 重置索引
        if not df_tmp.empty:
            pe_list.append(df_tmp.loc[0,'peTTM'])
            pb_list.append(df_tmp.loc[0,'pbMRQ'])
            ps_list.append(df_tmp.loc[0,'psTTM'])
    if len(pe_list) > 0:
        try :
            (pe,pb,ps) = calc_average(pe_list,pb_list,ps_list)
            return (date,round(pe, 2), round(pb, 2), round(ps,2) )
        except:
            return None
    else:
        return None


def combine_all_markt_stocks():
    """合并所有存在的股票数据"""
    paths = get_exists_stocks_path()
    df_list = []
    for path in tqdm(paths,desc='合并所有股票文件'):
        df_tmp = read_csv(path,code=str,date=str)
        df_tmp = df_tmp.loc[:,['date','code','peTTM','pbMRQ','psTTM']]
        df_list.append(df_tmp)
    df = pd.concat(df_list)
    return df

def get_all_market_pe_pb_date(df,date):
    df_date = df[df.date == date]
    pe_list = df_date['peTTM'].to_list()
    pb_list = df_date['pbMRQ'].to_list()
    ps_list = df_date['psTTM'].to_list()
    (pe,pb,ps) = calc_average(pe_list,pb_list,ps_list,method='mid')
    return (date,round(pe,2),round(pb,2),round(ps,2))

def get_all_market_pe_pb():
    """计算指定期间的全市场pe,pb"""
    flag = judge_update_stock(is_all=True)
    path = os.path.join(PATH_MARKET,'all_market_pe_pb.csv')
    end_date = datetime.date.today() - datetime.timedelta(1)
    end_date = end_date.strftime('%Y-%m-%d')
    if os.path.exists(path):
        df = read_csv(path,trade_date=str)
        if flag:
            updated_date = df.iloc[-1].trade_date
            updated_date = datetime.datetime.strptime(updated_date,'%Y-%m-%d')
            start_date = updated_date + datetime.timedelta(1)
            start_date = start_date.strftime('%Y-%m-%d')
            # df_tmp = get_index_pe_pb(start_date=start_date)
            df_tmp = calc_all_market_pe_pb(start_date,end_date)
            df = pd.concat([df, df_tmp])
    else:
        df = calc_all_market_pe_pb('1990-01-01', end_date)
        # df = get_index_pe_pb(start_date='1990-01-01')
    df.to_csv(path,index=False)
    return df


def calc_all_market_pe_pb(start_date,end_date):
    """计算指定期间的全市场pe,pb"""
    df = combine_all_markt_stocks()
    df['code'] = df['code'].apply(lambda x:re.sub('\D','',x)) # 去除股票代码中非数字项
    start = datetime.datetime.strptime(start_date,'%Y-%m-%d')
    end = datetime.datetime.strptime(end_date,'%Y-%m-%d')
    trade_date_list = get_trade_date()
    date_range = pd.date_range(start=start, end=end,freq="D")
    dates = [ date.strftime('%Y-%m-%d') for date in date_range if date.strftime('%Y-%m-%d') in trade_date_list ]
    args = [(df,date) for date in dates]
    result = []
    qbar = tqdm(dates)
    for date in qbar:
        qbar.set_description('计算%s全市场估值' % date)
        df_date = df[df.date == date]
        if not df_date.empty:
            pe_list = df_date['peTTM'].to_list()
            pb_list = df_date['pbMRQ'].to_list()
            ps_list = df_date['psTTM'].to_list()
            (pe,pb,ps) = calc_average(pe_list,pb_list,ps_list,method='mid')
            result.append([date,pe,pb,ps])
    df = pd.DataFrame(result,columns=['trade_date','PE','PB','PS'])
    return df

def all_market_pe_pb_legu():
    """乐咕全市场"""
    path_pe = './all_market/all_market_pe_legu.csv'
    path_pb = './all_market/all_market_pb_legu.csv'
    flag = judge_update_stock(is_all=True)
    if os.path.exists(path_pe):
        if flag:
            df_pe = ak.stock_a_ttm_lyr()
            df_pe.to_csv(path_pe,index=False)
        else:
            df_pe = read_csv(path_pe)
    else:
        df_pe = ak.stock_a_ttm_lyr()
        df_pe.to_csv(path_pe,index=False)
    if os.path.exists(path_pb):
        if flag:
            df_pb = ak.stock_a_all_pb()
            df_pb.to_csv(path_pb,index=False)
        else:
            df_pb = read_csv(path_pb)
    else:
        df_pb = ak.stock_a_all_pb()
        df_pb.to_csv(path_pb,index=False)
    pe_ratio = df_pe.iloc[-1].quantileInAllHistoryMiddlePeTtm * 100 # 历史百分位
    pb_ratio = df_pb.iloc[-1].quantileInAllHistoryMiddlePB * 100 # 历史百分位
    df_pe = df_pe.loc[:,['date','middlePETTM']]
    df_pb = df_pb.loc[:,['date','middlePB']]
    df_pe.columns = ['trade_date','PE']
    df_pb.columns = ['trade_date','PB']
    df = pd.merge(df_pe,df_pb,on='trade_date')
    date = df.iloc[-1].trade_date
    # (pe_ratio,pb_ratio) = calc_ratio(df,'PE','PB')
    title='%s全市场中位数PE、PB     当前PE百分位：%.2f,当前PB百分位：%.2f' % (date,pe_ratio,pb_ratio)
    plot(df,title)
    write_update_date(is_all=True)
    

def all_market_value(years=None):
    """全市场pe,pb"""
    get_all_k_data() # 更新所有股票数据
    df = get_all_market_pe_pb()
    if years:
        df = filter_recent_years(df,years)
    else:
        df = df[df.PE > 0]
    (pe_ratio,pb_ratio) = calc_ratio(df,'PE','PB')
    df = df.reset_index(drop=True)
    min_pe = df['PE'].min()
    max_pe = df['PE'].max()
    mid_pe = df['PE'].median()
    desc_pe = calc_state(pe_ratio)
    min_pb = df['PB'].min()
    max_pb = df['PB'].max()
    mid_pb = df['PB'].median()
    desc_pb = calc_state(pb_ratio)
    init_date = df.loc[0,'trade_date']
    columns=[
        '日期','PE','PE百分位','PE估值','PB','PB百分位','PB估值',
        'PE最小值','PE最大值','PE中位值','PB最小值','PB最大值','PB中位值',
        '起始日期'
    ]
    df = pd.DataFrame(
        [df.iloc[-1].trade_date,
        df.iloc[-1].PE,
        '%.2f' % pe_ratio,
        desc_pe,
        df.iloc[-1].PB,
        '%.2f' % pb_ratio,
        desc_pb,
        min_pe,max_pe,mid_pe,
        min_pb,max_pb,mid_pb,
        init_date]
    )
    df = df.T
    df.columns = columns
    write_update_date(is_all=True)
    fmt = 'fancy_grid'
    print(tabulate(df, headers='keys', tablefmt=fmt))

def calc_index_pe_pb(date,code=None):
    """计算指数一天的pe,pb,ps"""
    # print('计算指数%s在日期%s的估值' % (code,date))
    if code:
        stocks = get_stocks(code,date)
    else:
        stocks = get_all_stocks()
    pe_pb = get_index_pe_pb_date(date, stocks)
    return pe_pb


def get_index_pe_pb(start_date=None, end_date=None,code=None):
    '''指数历史PE_PB'''
    if code:
        init_date = get_security_info(code).iloc[0,-1] # 获取指数信息中的publish_date
    else:
        init_date = '1990-01-01'
    pe_list = []
    pb_list = []
    ps_list = []
    day_list = []
    if start_date is None:
        start_date = init_date
    if end_date is None:
        end_date = datetime.date.today() - datetime.timedelta(1) #如果有误，请删除#号 ，获取的是前一天的数据
        end_date = datetime.date.strftime(end_date, '%Y-%m-%d')

    trade_date_list = get_trade_date()
    date_range = pd.date_range(start=start_date, end=datetime.date.today()-datetime.timedelta(1),freq="D")#交易日
    if code:
        args = [(day.strftime('%Y-%m-%d'),code) for day in date_range if day.strftime('%Y-%m-%d') in trade_date_list]
    else:
        args = [(day.strftime('%Y-%m-%d'),None) for day in date_range if day.strftime('%Y-%m-%d') in trade_date_list]


    pool = Pool()
    result = pool.istarmap(calc_index_pe_pb,args)
    result_list = []
    pbar = tqdm(result,total=len(args))
    for _ in pbar:
        result_list.append(_)
    pool.close()
    pool.join()

    # result_list = []
    # for arg in args:
    #     pe_pb = calc_index_pe_pb(arg[0],arg[1])
    #     result_list.append(pe_pb)
    

    result = [i for i in result_list if i]
    df = pd.DataFrame(result,columns=['trade_date','PE','PB','PS'])
    df = df.sort_values(by='trade_date')
    return df


def get_hs_data(index_list):
    '''增量更新沪深指数估值数据'''
    for code in index_list:
        print(u'正在计算:', code)
        path = os.path.join(PATH_INDEX,'%s_pe_pb.csv' % code)
        if os.path.exists(path): #增量更新
            df_pe_pb = pd.read_csv(path,
                                   converters={'trade_date': str},
                                   index_col=False)
            start_date = datetime.datetime.strptime(df_pe_pb.iloc[-1].trade_date,
                                           '%Y-%m-%d') + datetime.timedelta(1)
            start_date = start_date.strftime('%Y-%m-%d')
            df_pe_pb = pd.concat([df_pe_pb, get_index_pe_pb(code=code, start_date=start_date)])
        else: #初次计算
            print('init')
            df_pe_pb = get_index_pe_pb(code=code)
        if not df_pe_pb.empty:
            df_pe_pb.to_csv(path, index=None)
            print('已更新%s-pepbps数据' % code)
        else:
            print('未更新%s-pepbps数据' % code)

def filter_recent_years(df, year_num):
    """筛选近几年的数据
    df:指数估值表
    year_num:年数
    """
    end_date = datetime.datetime.strptime(df.iloc[-1].trade_date, '%Y-%m-%d')
    start_date = end_date - datetime.timedelta(365 * year_num)
    df = df[df['trade_date'] > start_date.strftime('%Y-%m-%d')]
    return df

def calc_state(data):
    if data < 10.0:
        return u'极度低估'
    elif 10 <= data and data < 20:
        return u'低估'
    elif 20 <= data and data < 40:
        return u'正常偏低'
    elif 40 <= data and data < 60:
        return u'正常'
    elif 60 <= data and data < 80:
        return u'正常偏高'
    elif 80 <= data and data < 90:
        return u'高估'
    elif 90 <= data:
        return u'极度高估'

def calc_ratio(df,*args):
    """计算比率"""
    ratio_list = []
    for arg in args:
        ratio = len(df[df[arg] < df.iloc[-1][arg]]) / float(len(df)) * 100
        ratio_list.append(ratio)
    return tuple(ratio_list)

def pe_pb_analysis(index_list=['000300', '000905']):
    '''PE_PB分析'''
    all_index = get_security_info()
    all_index=all_index.set_index('index_code')
    hk_idx_name = {'hscei': u'国企指数', 'hsi': u'恒生指数'}
    pe_results = []
    pe_code_list = []
    pb_results = []
    pb_code_list = []
    ps_results = []
    ps_code_list = []
    #沪深
    for code in index_list:
        path = os.path.join(PATH_INDEX,'%s_pe_pb.csv' % code)
        if not os.path.exists(path): #增量更新
            continue
        index_name = all_index.loc[code, 'display_name']
        # df_pe_pb = read_csv(path,trade_date=str)
        df_pe_pb = pd.read_csv(path,
                               index_col=None,
                               converters={'trade_date': str})
        df_pe_pb = filter_recent_years(df_pe_pb, 10)
        if len(df_pe_pb) < 250 * 3: #每年250个交易日,小于3年不具有参考价值
            continue
        (pe_ratio, pb_ratio, ps_ratio) = calc_ratio(df_pe_pb,'PE','PB','PS')
        pe_results.append([
            index_name, df_pe_pb.iloc[-1].PE,
            '%.2f' % pe_ratio,
            calc_state(pe_ratio),
            min(df_pe_pb.PE),
            max(df_pe_pb.PE),
            '%.2f' % np.median(df_pe_pb.PE),
            '%.2f' % np.std(df_pe_pb.PE), df_pe_pb.iloc[0].trade_date
        ])
        pb_results.append([
            index_name, df_pe_pb.iloc[-1].PB,
            '%.2f' % pb_ratio,
            calc_state(pb_ratio),
            min(df_pe_pb.PB),
            max(df_pe_pb.PB),
            '%.2f' % np.median(df_pe_pb.PB),
            '%.2f' % np.std(df_pe_pb.PB), df_pe_pb.iloc[0].trade_date
        ])
        ps_results.append([
            index_name, df_pe_pb.iloc[-1].PS,
            '%.2f' % ps_ratio,
            calc_state(ps_ratio),
            min(df_pe_pb.PS),
            max(df_pe_pb.PS),
            '%.2f' % np.median(df_pe_pb.PS),
            '%.2f' % np.std(df_pe_pb.PS), df_pe_pb.iloc[0].trade_date
        ])
        pe_code_list.append(code)
        pb_code_list.append(code)
        ps_code_list.append(code)
    date_str = df_pe_pb.iloc[-1].trade_date
    # 恒生指数、恒生国企指数
    for code,index_name in hk_idx_name.items():
        df_pe_pb = get_hk_pe_pb(code)
        df_pe_pb = filter_recent_years(df_pe_pb,10)
        (pe_ratio,) = calc_ratio(df_pe_pb,'PE')
        pe_results.append([
            index_name,df_pe_pb.iloc[-1].PE,
            '%.2f' % pe_ratio,
            calc_state(pe_ratio),
            min(df_pe_pb.PE),
            max(df_pe_pb.PE),
            '%.2f' % df_pe_pb['PE'].median(),
            '%.2f' % df_pe_pb['PE'].std(),df_pe_pb.iloc[0].trade_date
        ])
    pe_columns = [
        u'名称', u'当前PE', u'百分位(%)', u'估值状态', u'最小', u'最大', u'中位数', u'标准差',
        u'起始日期'
    ]
    pe_df = pd.DataFrame(data=pe_results,
                         index=pe_code_list + list(hk_idx_name.keys()),
                         columns=pe_columns)
    pe_df.index.name = date_str
    pb_columns = [
        u'名称', u'当前PB', u'百分位(%)', u'估值状态', u'最小', u'最大', u'中位数', u'标准差',
        u'起始日期'
    ]
    pb_df = pd.DataFrame(data=pb_results,
                         index=pb_code_list ,
                         columns=pb_columns)
    pb_df.index.name = date_str
    ps_columns = [
        u'名称', u'当前PS', u'百分位(%)', u'估值状态', u'最小', u'最大', u'中位数', u'标准差',
        u'起始日期'
    ]
    ps_df = pd.DataFrame(data=ps_results,
                         index=ps_code_list,
                         columns=ps_columns)
    ps_df.index.name = date_str
    pe_df = pe_df.apply(pd.to_numeric, errors='ignore')
    pb_df = pb_df.apply(pd.to_numeric, errors='ignore')
    ps_df = ps_df.apply(pd.to_numeric, errors='ignore')
    return (pe_df.sort_values(by='百分位(%)', ascending=True),
            pb_df.sort_values(by='百分位(%)', ascending=True),
            ps_df.sort_values(by='百分位(%)', ascending=True),
            )

def write_update_date(is_all=False):
    with open(PATH_CONFIG,'rb') as f:
        json_str = json.load(f)
        if is_all:
            json_str['update_date_all'] = str(datetime.date.today())
        else:
            json_str['update_date'] = str(datetime.date.today())
    with open(PATH_CONFIG,'w') as f:
        json.dump(json_str,f,indent=2)

def index_customer_value():
    """将自选的指数生成pe\pb\ps估值列表"""
    stock_list = get_all_index_stocks(INDEX_LIST)
    update_stock_data(stock_list,use_flag=True)
    get_hs_data(INDEX_LIST)
    pe,pb,ps = pe_pb_analysis(INDEX_LIST)
    # 写入更新时间
    write_update_date()
    fmt = 'fancy_grid'
    print(tabulate(pe, headers='keys', tablefmt=fmt))
    print(tabulate(pb, headers='keys', tablefmt=fmt))
    print(tabulate(ps, headers='keys', tablefmt=fmt))

def stock_info(is_all=False):
    """获取股票信息"""
    flag = judge_update_stock(is_all)
    path = os.path.join(PATH_INFO,'stock_info.csv')
    if os.path.exists(path) and not flag:
        df = read_csv(path,code=str)
    else:
        df = ak.stock_info_a_code_name()
        df.to_csv(path,index=False)
    return df

def get_all_k_data():
    """更新A股所有股票数据"""
    df = stock_info(is_all=True)
    stocks = df['code'].to_list()
    update_stock_data(stocks,is_all=True,use_flag=True)

def request_hk(url,path):
    """获取恒生指数与恒生国企指数PE"""
    response = requests.get(url)
    df = pd.DataFrame(response.json())
    df = df.fillna('')
    if not df.empty:
        df.columns = ['trade_date','PE','close']
        df.to_csv(path,index=False)
    else:
        df = None
    return df

def get_hk_pe_pb(code):
    """获取恒生指数与恒生国企指数PE"""
    if code == 'hsi':
        url = 'https://eniu.com/chart/peindex/hkhsi/t/all'
        path = os.path.join(PATH_INDEX,'hsi_pe_pb.csv')
    elif code == 'hscei':
        url = 'https://eniu.com/chart/peindex/hkhscei/t/all'
        path = os.path.join(PATH_INDEX,'hscei_pe_pb.csv')
    else:
        return None
    if os.path.exists(path):
        df = read_csv(path)
        data_date = df.iloc[-1].trade_date
        yestoday = datetime.date.today() - datetime.timedelta(days=1)
        if data_date>=str(yestoday):
            pass
        else:
            df = request_hk(url,path)
    else:
        df = request_hk(url,path)
    return df

def plot_pe_pb(df,title,**kw):
    """绘图主程序"""
    line1 = go.Scatter(x=df.trade_date,y=df.PE,mode='lines',name='PE',
                       hovertemplate='<b>日期</b>:%{x|%Y-%m-%d}<br><b>PE</b>:%{y}',
                      line_color='#555555') #EF553B
    line2 = go.Scatter(x=df.trade_date,y=df.PB,mode='lines',name='PB',yaxis='y2',
                      hovertemplate='<b>日期</b>:%{x|%Y-%m-%d}<br><b>PB</b>:%{y}',
                      line_dash='dot',line_color='#778AAE')
    fig = go.Figure([line1,line2])
    relation = {
        'pe_high': {'name':'90% PE','color':'#DC3912','yaxis':'y','dash':None},
        'pe_low': {'name':'10% PE','color':'#66AA00','yaxis':'y','dash':None},
        'pe_mid': {'name':'50% PE','color':'#FF9900','yaxis':'y','dash':None},
        'pb_high': {'name':'90% PB','color':'#DC3912','yaxis':'y2','dash':'dot'},
        'pb_low': {'name':'10% PB','color':'#66AA00','yaxis':'y2','dash':'dot'},
        'pb_mid': {'name':'50% PB','color':'#FF9900','yaxis':'y2','dash':'dot'},
    }
    if kw:
        for k,v in relation.items():
            if k in kw.keys():
                line = go.Scatter(x=df.trade_date,y=[kw[k]] * len(df) ,mode='lines',name=v['name'],line_color=v['color'],yaxis=v['yaxis'],line_dash=v['dash'])
                fig.add_trace(line)
    else:
        fig.update_traces(line_color='#EF553B',selector={'name':'PE'})

    fig.update_layout(title=title,
                      yaxis_title='PE',
                      yaxis_dtick=10,
                      yaxis2=dict(title='PB',overlaying='y',side='right',dtick=1),
                      xaxis_title='日期',
                      # hovermode="y unified",
                      # xaxis_dtick = 'Y1'
                     )
    fig.show()

def plot_all_market(start_date='1995-01-01'):
    """绘制全市场估值趋势图"""
    path = os.path.join(PATH_MARKET,'all_market_pe_pb.csv')
    df = pd.read_csv(path)
    df = df[df.trade_date >= start_date]
    plot_pe_pb(df,'全市场估值')

def plot(df,title):
    """绘制基础图形"""
    pe_high = df['PE'].quantile(0.9)
    pe_low = df['PE'].quantile(0.1)
    pe_mid = df['PE'].quantile(0.5)
    pb_high = df['PB'].quantile(0.9)
    pb_low = df['PB'].quantile(0.1)
    pb_mid = df['PB'].quantile(0.5)
    plot_pe_pb(df,title,pe_high=pe_high,pe_low=pe_low,pe_mid=pe_mid,pb_high=pb_high,pb_low=pb_low,pb_mid=pb_mid)


def plot_index(code,start_date='2000-01-01'):
    """绘制指数估计趋势图"""
    path = os.path.join(PATH_INDEX,'%s_pe_pb.csv' % code)
    df = pd.read_csv(path)
    df = df[df.trade_date >= start_date]
    info = get_security_info(code)
    name = info.iloc[0].display_name
    (pe_ratio, pb_ratio) = calc_ratio(df,'PE','PB')
    last_date = df.iloc[-1].trade_date
    title = '%s %s估值   %s当前PE百分位:%.2f,当前PB百分位:%.2f' % (name,code,last_date,pe_ratio,pb_ratio)
    plot(df,title)

def plot_stock(code,start_date='1995-01-01'):
    """绘制个股的估值图"""
    path = os.path.join(PATH_STOCK,'%s_indicator.csv' % code)
    df = pd.read_csv(path)
    df = df.loc[:,['date','peTTM','pbMRQ','psTTM']]
    df.columns = ['trade_date','PE','PB','PS']
    if code[:2] =='hk':
        name = ''
    else:
        info = stock_info()
        info = info[info.code==code]
        name = info.iloc[0]['name']
    df = df[df.trade_date >= start_date]
    (pe_ratio, pb_ratio) = calc_ratio(df,'PE','PB')
    last_date = df.iloc[-1].trade_date
    title = '%s %s估值   %s当前PE百分位:%.2f,当前PB百分位:%.2f' % (name,code,last_date,pe_ratio,pb_ratio)
    plot(df,title)

def check_dir():
    path_list = [PATH_INDEX,PATH_INFO,PATH_STOCK,PATH_MARKET,PATH_WEIGHT]
    for path in path_list:
        if not os.path.exists(path):
            os.mkdir(path)

def main():
    if len(sys.argv)<2:
        print('需要至少1个参数\r\nmarket：全市场估计\r\nindex:指数估值\r\nstock:个股估值')
    else:
        check_dir()
        if sys.argv[1] == 'market':
            if len(sys.argv)<3:
                all_market_pe_pb_legu()
                # all_market_value()
                # plot_all_market()
                # print("输入python calc.py market 2015-01-01\r\n将从指定日期开始计算\r\n否则默认图形从1995-01-01开始绘制")
            else:
                all_market_pe_pb_legu()
                # all_market_value()
                # plot_all_market(sys.argv[2])
        elif sys.argv[1] == 'index':
            if len(sys.argv)<3:
                index_customer_value()
                print("需要输入指数代码，如:\r\npython calc.py index 000827")
            elif len(sys.argv)==3:
                stocks = get_all_index_stocks([sys.argv[2]])
                update_stock_data(stocks)
                get_hs_data([sys.argv[2]])
                plot_index(sys.argv[2])
            else:
                stocks = get_all_index_stocks([sys.argv[2]])
                update_stock_data(stocks)
                get_hs_data([sys.argv[2]])
                plot_index(sys.argv[2],sys.argv[3])
        elif sys.argv[1] == 'stock':
            if len(sys.argv)<3:
                print("需要输入股票代码，如:\r\npython calc.py stock 000002")
            elif len(sys.argv)==3:
                update_stock_data([sys.argv[2]])
                plot_stock(sys.argv[2])
            else:
                update_stock_data([sys.argv[2]])
                plot_stock(sys.argv[2],sys.argv[3])
        else:
            print('参数错误')

if __name__ == "__main__":
    main()
    # get_hk_stock('hk09988')
    # update_stock_data('hk09988')
    # all_market_pe_pb_legu()
    # index_customer_value()
    # all_market_value()
    # check_dir()
