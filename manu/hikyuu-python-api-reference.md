# Hikyuu Python API 完整参考

**文档创建时间**: 2025年8月25日  
**分析工具**: DeepWiki MCP Server  
**框架版本**: fasiondog/hikyuu (GitHub)

## 目录

1. [核心类](#核心类)
2. [技术指标函数](#技术指标函数)
3. [交易系统组件](#交易系统组件)
4. [投资组合组件](#投资组合组件)
5. [工厂函数](#工厂函数)
6. [工具函数](#工具函数)
7. [枚举和常量](#枚举和常量)

## 核心类

### StockManager - 证券管理类

全局单例，管理所有股票数据和市场信息。

#### 属性
- `instance()` - 获取全局实例

#### 方法
- `get_market_list()` - 获取市场简称列表
- `get_market_info(market)` - 获取指定市场的详细信息
- `get_stock(querystr)` - 根据"市场简称证券代码"获取证券实例
- `get_stock_list([filter])` - 获取证券列表，可选择过滤函数
- `get_block(category, name)` - 获取预定义的板块
- `get_trading_calendar(query)` - 获取交易日历
- `__getitem__(market_code)` - 通过索引获取股票
- `__len__()` - 获取股票总数

### Stock - 股票类

代表一个证券对象，包含证券的基本信息和数据获取方法。

#### 属性
- `id` - 内部id，通常用作map的键值
- `market` - 所属市场简称
- `code` - 证券代码
- `market_code` - 市场简称+证券代码，例如 "sh000001"
- `name` - 证券名称
- `type` - 证券类型
- `valid` - 该证券当前是否有效
- `start_datetime` - 证券起始日期
- `last_datetime` - 证券最后日期
- `tick` - 最小跳动量
- `tick_value` - 最小跳动量价值
- `unit` - 每单位价值 = tickValue / tick
- `precision` - 价格精度
- `atom` - 最小交易数量，同 min_trade_number
- `min_trade_number` - 最小交易数量
- `max_trade_number` - 最大交易数量

#### 方法
- `is_null()` - 判断是否为Null对象
- `is_buffer(ktype)` - 判断指定类型的K线数据是否被缓存
- `get_kdata(query)` - 获取满足查询条件的K线数据，返回 KData 对象
- `get_timeline_list(query)` - 获取分时线数据
- `get_trans_list(query)` - 获取历史分笔数据
- `get_count(ktype=Query.DAY)` - 获取不同类型K线数据量
- `get_market_value(date, ktype)` - 获取指定时刻的市值
- `get_krecord(pos, ktype=Query.DAY)` - 获取指定索引或日期的K线数据记录
- `get_krecord_list(start, end, ktype)` - 获取K线记录列表 [start, end)
- `get_datetime_list(query)` - 获取日期列表
- `get_weight(start, end)` - 获取指定时间段内的权息信息
- `get_finance_info()` - 获取当前财务信息
- `get_history_finance()` - 获取所有历史财务信息列表
- `set_krecord_list(krecord_list, ktype=Query.DAY)` - 设置内存中的KRecordList
- `realtime_update(krecord, ktype=Query.DAY)` - 更新内存缓存中的K线数据
- `load_kdata_to_buffer(ktype)` - 将指定类别的K线数据加载至内存缓存

### KData - K线数据集合类

代表K线数据集合，通过 Stock.get_kdata() 方法获取。

#### 属性
- `stock` - 对应的 Stock 对象
- `query` - 对应的 KQuery 对象
- `size` - K线记录的数量

**注意**: KData对象通过索引访问具体的KRecord来获取时间信息，如 `kdata[0].datetime` 和 `kdata[-1].datetime`

#### 方法
- `get_datetime(pos)` - 获取指定索引的日期
- `get_open(pos)` - 获取指定索引的开盘价
- `get_high(pos)` - 获取指定索引的最高价
- `get_low(pos)` - 获取指定索引的最低价
- `get_close(pos)` - 获取指定索引的收盘价
- `get_vol(pos)` - 获取指定索引的成交量
- `get_amo(pos)` - 获取指定索引的成交金额
- `get_krecord(pos)` - 获取指定索引的 KRecord 对象
- `get_krecord_list()` - 获取所有 KRecord 对象的列表
- `empty()` - 判断KData是否为空
- `open()` - 返回开盘价的 Indicator 对象
- `high()` - 返回最高价的 Indicator 对象
- `low()` - 返回最低价的 Indicator 对象
- `close()` - 返回收盘价的 Indicator 对象
- `vol()` - 返回成交量的 Indicator 对象
- `amo()` - 返回成交金额的 Indicator 对象
- `plot()` - 绘制K线图
- `__getitem__(pos)` - 支持通过索引访问 KRecord
- `__len__()` - 返回K线记录的数量
- `__iter__()` - 支持迭代K线记录

### Query - 查询条件类

用于构建K线数据查询条件。

#### 属性
- `start` - 起始索引
- `end` - 结束索引
- `start_datetime` - 起始日期
- `end_datetime` - 结束日期
- `query_type` - 查询方式
- `ktype` - 查询的K线类型
- `recover_type` - 复权类型

#### 枚举值
**K线类型 (KType)**:
- `Query.DAY` - 日线类型
- `Query.WEEK` - 周线类型
- `Query.MONTH` - 月线类型
- `Query.QUARTER` - 季线类型
- `Query.HALFYEAR` - 半年线类型
- `Query.YEAR` - 年线类型
- `Query.MIN` - 1分钟线类型
- `Query.MIN5` - 5分钟线类型
- `Query.MIN15` - 15分钟线类型
- `Query.MIN30` - 30分钟线类型
- `Query.MIN60` - 60分钟线类型
- `Query.HOUR2` - 2小时线类型
- `Query.HOUR4` - 4小时线类型
- `Query.HOUR6` - 6小时线类型
- `Query.HOUR12` - 12小时线类型

**复权类型 (RecoverType)**:
- `Query.NO_RECOVER` - 不复权
- `Query.FORWARD` - 前复权
- `Query.BACKWARD` - 后复权
- `Query.EQUAL_FORWARD` - 等比前复权
- `Query.EQUAL_BACKWARD` - 等比后复权

### TradeManager - 交易管理类

模拟账户进行交易管理，包括资金、持仓和交易记录的管理。

#### 属性
- `name` - 账户名称
- `cost_func` - 交易成本算法
- `init_cash` - 初始资金（只读）
- `current_cash` - 当前资金（只读）
- `init_datetime` - 账户建立日期（只读）
- `first_datetime` - 第一笔买入交易发生日期（只读）
- `last_datetime` - 最后一笔交易日期（只读）
- `precision` - 价格精度（只读）
- `broker_last_datetime` - 实际开始订单代理操作的时刻

#### 方法
**参数管理**:
- `get_param(name)` - 获取指定的参数
- `set_param(name, value)` - 设置参数
- `have_param(name)` - 判断是否存在指定参数

**基本操作**:
- `reset()` - 复位，清空交易、持仓记录
- `clone()` - 克隆（深复制）实例
- `reg_broker(broker)` - 注册订单代理
- `clear_broker()` - 清空所有已注册订单代理

**持仓查询**:
- `have(stock)` - 当前是否持有指定的证券
- `get_stock_num()` - 当前持有的证券种类数量
- `get_short_stock_num()` - 获取当前融券持有的证券种类数量
- `get_hold_num(datetime, stock)` - 获取指定时刻指定证券的持有数量
- `get_short_hold_num(datetime, stock)` - 获取指定时刻指定证券的融券持有数量
- `get_margin_rate(datetime, stock)` - 获取保证金比率

**交易记录**:
- `get_trade_list([start, end])` - 获取交易记录
- `get_position_list()` - 获取当前全部持仓记录
- `get_history_position_list()` - 获取全部历史持仓记录
- `get_position(date, stock)` - 获取指定日期指定证券的持仓记录

**成本计算**:
- `get_buy_cost(datetime, stock, price, num)` - 计算买入成本
- `get_sell_cost(datetime, stock, price, num)` - 计算卖出成本

**交易操作**:
- `buy(datetime, stock, real_price, number, stoploss, goal_price, plan_price, from, remark)` - 买入操作
- `sell(datetime, stock, real_price, number, stoploss, goal_price, plan_price, from, remark)` - 卖出操作
- `sell_short(datetime, stock, real_price, number, stoploss, goal_price, plan_price, from, remark)` - 融券卖出操作
- `buy_short(datetime, stock, real_price, number, stoploss, goal_price, plan_price, from, remark)` - 融券买入操作

**融资融券**:
- `borrow_cash(datetime, cash)` - 融资操作
- `return_cash(datetime, cash)` - 归还融资
- `borrow_stock(datetime, stock, price, number)` - 融券操作
- `return_stock(datetime, stock, price, number)` - 归还融券

**资金操作**:
- `checkin(datetime, cash)` - 存入资金
- `checkout(datetime, cash)` - 取出资金
- `checkin_stock(datetime, stock, price, number)` - 存入证券
- `checkout_stock(datetime, stock, price, number)` - 取出证券

**绩效分析**:
- `get_funds(datetime, ktype)` - 获取指定时刻的资金情况
- `get_funds_list(dates, ktype)` - 获取指定日期列表的资金情况
- `get_funds_curve(dates, ktype)` - 获取资金曲线
- `get_profit_curve(dates, ktype)` - 获取利润曲线
- `get_profit_cum_change_curve(dates, ktype)` - 获取累计收益率曲线
- `get_base_assets_curve(dates, ktype)` - 获取基准资产曲线
- `get_performance(datetime, ktype)` - 获取绩效统计
- `performance(query)` - 打印绩效报告

**数据导出**:
- `tocsv(path)` - 导出交易记录、持仓和绩效曲线到 CSV 文件
- `fetch_asset_info_from_broker(broker, datetime)` - 从订单代理获取资产信息

### System - 交易系统类

实现完整交易策略的核心组件。

#### 属性
- `name` - 系统名称
- `tm` - 交易管理对象
- `mm` - 资金管理策略
- `ev` - 市场环境判断策略
- `cn` - 系统有效条件
- `sg` - 信号指示器
- `st` - 止损策略
- `tp` - 止盈策略
- `pg` - 盈利目标策略
- `sp` - 移滑价差算法

#### 方法
- `get_param(name)` - 获取指定参数的值
- `set_param(name, value)` - 设置参数的值
- `have_param(name)` - 判断是否存在指定参数
- `set_not_shared_all()` - 将所有组件设置为非共享
- `get_stock()` - 获取关联的证券
- `get_trade_record_list()` - 获取实际执行的交易记录
- `get_buy_trade_request()` - 获取买入请求
- `get_sell_trade_request()` - 获取卖出请求
- `get_sell_short_trade_request()` - 获取卖空请求
- `get_buy_short_trade_request()` - 获取买空请求
- `run(stock, query)` - 运行系统回测
- `performance(query)` - 获取系统绩效

### Portfolio - 投资组合类

用于实现多标的、多策略的投资组合管理。

#### 属性
- `name` - 组合名称
- `query` - 查询条件
- `tm` - 交易管理对象
- `se` - 交易对象选择算法
- `af` - 资产分配算法
- `real_sys_list` - 实际运行系统列表

#### 方法
- `set_param(name, value)` - 设置参数
- `get_param(name)` - 获取参数
- `run(query)` - 运行投资组合回测
- `performance(query)` - 获取组合绩效

### Strategy - 策略类

策略运行时的主控制器，提供事件驱动架构。

#### 属性
- `name` - 策略名称
- `running` - 当前运行状态（只读）
- `context` - 策略上下文（只读）
- `tm` - 关联的交易管理实例
- `sp` - 移滑价差算法
- `is_backtesting` - 回测状态（只读）

#### 方法
- `start(auto_recieve_spot=True)` - 启动策略执行
- `stop()` - 停止策略执行
- `pause()` - 暂停策略执行
- `resume()` - 恢复策略执行
- `now()` - 获取当前时间
- `today()` - 获取当前日期
- `get_kdata(stock, query)` - 获取K线数据
- `get_last_kdata(stock, n, ktype)` - 获取最新N条K线数据
- `buy(stock, price, num)` - 买入操作
- `sell(stock, price, num)` - 卖出操作
- `on_change(func)` - 设置证券数据更新回调通知
- `on_received_spot(func)` - 设置证券数据更新通知回调
- `run_daily(func, time_delta, market, ignore_market)` - 设置日内循环执行回调
- `run_daily_at(func, time_point, ignore_market)` - 设置每日定点执行回调

### Indicator - 技术指标类

技术指标的基础类，支持各种数学和逻辑运算。

#### 属性
- `name` - 指标名称
- `size` - 指标长度
- `discard` - 抛弃的数据量
- `result_num` - 结果集数量

#### 方法
- `get_result_as_price_list()` - 获取结果集转换为价格列表
- `get_result(result_index)` - 获取指定结果集的数据（用于MACD等多结果指标）
- `set_context(stock, query)` - 设置指标上下文
- `get_context()` - 获取指标上下文
- `plot()` - 绘制指标
- `to_np()` - 转换为numpy数组
- `__getitem__(pos)` - 通过索引访问指标值
- `__len__()` - 获取指标长度
- `__iter__()` - 支持迭代

**API使用注意事项**:
- 对于MACD等多结果指标，应使用 `get_result(index)` 方法而非直接索引访问
- 在访问指标值前，建议检查指标长度: `if len(indicator) > 0:`

### Datetime - 日期时间类

处理日期时间的类。

#### 构造函数
- `Datetime()` - 默认构造函数
- `Datetime(year, month, day)` - 指定年月日
- `Datetime(year, month, day, hour, minute, second)` - 指定年月日时分秒
- `Datetime(datetime_int)` - 从整数构造

#### 静态方法
- `Datetime.now()` - 获取当前时间
- `Datetime.today()` - 获取今天日期
- `Datetime.min()` - 获取最小日期
- `Datetime.max()` - 获取最大日期

#### 属性
- `year` - 年
- `month` - 月
- `day` - 日
- `hour` - 时
- `minute` - 分
- `second` - 秒

### TimeDelta - 时间差类

表示时间间隔。

#### 构造函数
- `TimeDelta(days=0, hours=0, minutes=0, seconds=0)` - 指定时间差

#### 属性
- `days` - 天数
- `hours` - 小时数
- `minutes` - 分钟数
- `seconds` - 秒数

## 技术指标函数

### 行情指标

#### 基础数据
- `KDATA()` - 包装 KData 成 Indicator
- `KDATA_PART()` - 根据字符串选择返回指标
- `OPEN()` - 开盘价指标
- `HIGH()` - 最高价指标
- `LOW()` - 最低价指标
- `CLOSE()` - 收盘价指标
- `AMO()` - 成交金额指标
- `VOL()` - 成交量指标

#### 复权函数
- `RECOVER_FORWARD()` - 前复权
- `RECOVER_BACKWARD()` - 后复权
- `RECOVER_EQUAL_FORWARD()` - 等比前复权
- `RECOVER_EQUAL_BACKWARD()` - 等比后复权

#### 财务和市值
- `FINANCE([kdata, ix, name])` - 获取历史财务信息
- `HSL()` - 换手率
- `CAPITAL()` - 流通盘
- `ZONGGUBEN()` - 总股本

#### 分时数据
- `TIMELINE()` - 分时价格
- `TIMELINEVOL()` - 分时成交量

#### 债券指标
- `ZHBOND10([data, default_val])` - 10年期中国国债收益率

### 大盘指标

- `ADVANCE()` - 上涨家数
- `DECLINE()` - 下跌家数
- `INDEXO()` - 大盘开盘价
- `INDEXH()` - 大盘最高价
- `INDEXL()` - 大盘最低价
- `INDEXC()` - 大盘收盘价
- `INDEXA()` - 大盘成交金额
- `INDEXV()` - 大盘成交量
- `INDEXADV()` - 通达信大盘上涨家数
- `INDEXDEC()` - 通达信大盘下跌家数

### 移动平均指标

- `MA(data, n)` - 简单移动平均
- `EMA(data, n)` - 指数移动平均
- `SMA(data, n, m)` - 修正移动平均
- `WMA(data, n)` - 加权移动平均
- `AMA(data, n)` - 自适应移动平均
- `DEMA(data, n)` - 双重指数移动平均
- `TEMA(data, n)` - 三重指数移动平均
- `TRIMA(data, n)` - 三角移动平均
- `VIDYA(data, n, min_period)` - 可变指数动态平均

### 技术分析指标

#### 趋势指标
- `MACD(data, fast_n, slow_n, signal_n)` - 指数平滑异同移动平均线
- `PPO(data, fast_n, slow_n, signal_n)` - 价格震荡百分比指标
- `DIFF(data)` - 差分指标
- `SLOPE(data, n)` - 线性回归斜率

#### 震荡指标
- `RSI(data, n)` - 相对强弱指数
- `STOCHRSI(data, n, fastk_n, fastd_n, fastd_matype)` - 随机相对强弱指数
- `CCI(data, n)` - 商品通道指数
- `MFI(data, n)` - 资金流量指标
- `CMO(data, n)` - 钱德动量摆动指标
- `ROC(data, n)` - 变动率指标
- `ROCP(data, n)` - 变动率百分比
- `ROCR(data, n)` - 变动率比率
- `TRIX(data, n)` - 三重指数平滑平均线

#### 波动性指标
- `ATR(data, n)` - 平均真实波幅
- `NATR(data, n)` - 标准化平均真实波幅
- `TRANGE(data)` - 真实波幅
- `BOLL(data, n, p)` - 布林带
- `ENV(data, n, p)` - 包络线

#### 成交量指标
- `AD(data)` - 累积/派发线
- `ADOSC(data, fast_n, slow_n)` - 累积/派发震荡器
- `OBV(data)` - 能量潮
- `PVI(data)` - 正成交量指标
- `NVI(data)` - 负成交量指标

### 数学指标

#### 基本数学函数
- `ABS(data)` - 求绝对值
- `ACOS(data)` - 反余弦值
- `ASIN(data)` - 反正弦值
- `ATAN(data)` - 反正切值
- `COS(data)` - 余弦值
- `SIN(data)` - 正弦值
- `TAN(data)` - 正切值
- `EXP(data)` - e的X次幂
- `LN(data)` - 求自然对数
- `LOG(data)` - 以10为底的对数
- `SQRT(data)` - 开平方
- `POW(data, n)` - 乘幂
- `REVERSE(data)` - 求相反数
- `SGN(data)` - 求符号值

#### 取整函数
- `CEILING(data)` - 向上舍入取整
- `FLOOR(data)` - 向下舍入取整
- `INTPART(data)` - 取整（取得数据的整数部分）
- `ROUND(data, ndigits)` - 四舍五入
- `ROUNDUP(data, ndigits)` - 向上截取
- `ROUNDDOWN(data, ndigits)` - 向下截取

#### 最值函数
- `MAX(data)` - 最大值
- `MIN(data)` - 最小值
- `HHV(data, n)` - N日内最高价
- `LLV(data, n)` - N日内最低价
- `HHVBARS(data, n)` - 上一高点到当前的周期数
- `LLVBARS(data, n)` - 上一低点到当前的周期数

#### 数学运算
- `MOD(data, n)` - 取整后求模
- `SUM(data, n)` - 求和
- `COUNT(data, n)` - 计数

### 统计指标

- `AVEDEV(data, n)` - 平均绝对偏差
- `DEVSQ(data, n)` - 数据偏差平方和
- `STD(data, n)` - 估算标准差，同 STDEV
- `STDEV(data, n)` - 计算N周期内样本标准差
- `STDP(data, n)` - 总体标准差
- `VAR(data, n)` - 估算样本方差
- `VARP(data, n)` - 总体样本方差
- `CORR(ind, ref_ind, n)` - 样本相关系数与协方差
- `SPEARMAN(ind, ref_ind, n)` - Spearman相关系数
- `RANK(data, block, query, n, asc)` - 计算指标值在指定板块中的排名

### 逻辑算术函数

#### 逻辑判断
- `IF(condition, true_val, false_val)` - 根据条件求不同的值
- `AND(data1, data2)` - 逻辑与
- `OR(data1, data2)` - 逻辑或
- `NOT(data)` - 求逻辑非
- `BETWEEN(data, min_val, max_val)` - 介于两个数之间

#### 交叉函数
- `CROSS(data1, data2)` - 交叉函数
- `LONGCROSS(data1, data2, n)` - 两条线维持一定周期后交叉

#### 条件统计
- `EVERY(data, n)` - 条件在N周期一直存在
- `EXIST(data, n)` - 条件在N周期有存在
- `LAST(condition, n)` - 区间存在
- `DOWNNDAY(data, n)` - 连跌周期数
- `UPNDAY(data, n)` - 连涨周期数
- `NDAY(data, n)` - 连大

### 辅助类指标

- `ALIGN(data, ref, fill_null)` - 按指定的参考日期对齐
- `CYCLE()` - PF 调仓周期指标
- `CVAL(val, len)` - 创建指定长度的固定数值指标
- `CONTEXT()` - 独立上下文
- `DISCARD(data, n)` - 设置指标结果的丢弃数据量
- `DROPNA(data)` - 删除 nan 值
- `INBLOCK(block)` - 当前上下文证券是否在指定的板块中
- `ISNA(data)` - 判断是否为 nan 值
- `ISINF(data)` - 判断是否为 inf 值
- `JUMPDOWN(data, n)` - 向下跳变
- `JUMPUP(data, n)` - 向上跳变
- `LASTVALUE(data)` - 取输入指标最后值为常数
- `PRICELIST(data)` - 将 PriceList 或 Indicator 的结果集包装为 Indicator
- `REPLACE(data, old_val, new_val)` - 替换指标中指定值
- `RESULT(data, result_index)` - 返回指定指标中相应的结果集
- `SLICE(data, start, end, result_index)` - 获取某指标中指定范围的数据
- `WEAVE(data1, data2)` - 将两个 ind 的结果合并到一个 ind 中
- `WITHKTYPE(data, ktype)` - 将指标数据转换到指定的 K 线类型
- `ZSCORE(data, n)` - ZScore 标准化

### TA-Lib指标

Hikyuu集成了TA-Lib库，提供额外的技术指标：

- `TA_SMA(data, n)` - TA-Lib简单移动平均
- `TA_EMA(data, n)` - TA-Lib指数移动平均
- `TA_WMA(data, n)` - TA-Lib加权移动平均
- `TA_DEMA(data, n)` - TA-Lib双重指数移动平均
- `TA_TEMA(data, n)` - TA-Lib三重指数移动平均
- `TA_TRIMA(data, n)` - TA-Lib三角移动平均
- `TA_KAMA(data, n)` - TA-Lib自适应移动平均
- `TA_MAMA(data, fastlimit, slowlimit)` - TA-Lib自适应移动平均
- `TA_T3(data, n, vfactor)` - TA-Lib T3移动平均
- `TA_ADX(data, n)` - TA-Lib平均趋向指数
- `TA_ADXR(data, n)` - TA-Lib平均趋向指数评估
- `TA_APO(data, fast_n, slow_n, matype)` - TA-Lib绝对价格震荡器
- `TA_AROON(data, n)` - TA-Lib阿隆指标
- `TA_AROONOSC(data, n)` - TA-Lib阿隆震荡器
- `TA_BOP(data)` - TA-Lib均势指标
- `TA_CCI(data, n)` - TA-Lib商品通道指数
- `TA_DX(data, n)` - TA-Lib趋向指数
- `TA_MACD(data, fast_n, slow_n, signal_n)` - TA-Lib MACD
- `TA_MACDEXT(data, fast_n, fast_matype, slow_n, slow_matype, signal_n, signal_matype)` - TA-Lib扩展MACD
- `TA_MACDFIX(data, signal_n)` - TA-Lib固定12/26 MACD
- `TA_MFI(data, n)` - TA-Lib资金流量指标
- `TA_MINUS_DI(data, n)` - TA-Lib负向指标
- `TA_MINUS_DM(data, n)` - TA-Lib负向移动
- `TA_MOM(data, n)` - TA-Lib动量
- `TA_PLUS_DI(data, n)` - TA-Lib正向指标
- `TA_PLUS_DM(data, n)` - TA-Lib正向移动
- `TA_PPO(data, fast_n, slow_n, matype)` - TA-Lib价格震荡百分比
- `TA_ROC(data, n)` - TA-Lib变动率
- `TA_ROCP(data, n)` - TA-Lib变动率百分比
- `TA_ROCR(data, n)` - TA-Lib变动率比率
- `TA_ROCR100(data, n)` - TA-Lib变动率比率100倍
- `TA_RSI(data, n)` - TA-Lib相对强弱指数
- `TA_STOCH(data, fastk_n, slowk_n, slowk_matype, slowd_n, slowd_matype)` - TA-Lib随机指标
- `TA_STOCHF(data, fastk_n, fastd_n, fastd_matype)` - TA-Lib快速随机指标
- `TA_STOCHRSI(data, n, fastk_n, fastd_n, fastd_matype)` - TA-Lib随机RSI
- `TA_TRIX(data, n)` - TA-Lib三重指数平滑平均线
- `TA_ULTOSC(data, n1, n2, n3)` - TA-Lib终极震荡器
- `TA_WILLR(data, n)` - TA-Lib威廉%R
- `TA_AD(data)` - TA-Lib累积/派发线
- `TA_ADOSC(data, fast_n, slow_n)` - TA-Lib累积/派发震荡器
- `TA_OBV(data)` - TA-Lib能量潮
- `TA_ATR(data, n)` - TA-Lib平均真实波幅
- `TA_NATR(data, n)` - TA-Lib标准化平均真实波幅
- `TA_TRANGE(data)` - TA-Lib真实波幅
- `TA_BBANDS(data, n, nbdevup, nbdevdn, matype)` - TA-Lib布林带
- `TA_DEMA(data, n)` - TA-Lib双重指数移动平均
- `TA_EMA(data, n)` - TA-Lib指数移动平均
- `TA_HT_DCPERIOD(data)` - TA-Lib希尔伯特变换-主导周期
- `TA_HT_DCPHASE(data)` - TA-Lib希尔伯特变换-主导周期相位
- `TA_HT_PHASOR(data)` - TA-Lib希尔伯特变换-相量分量
- `TA_HT_SINE(data)` - TA-Lib希尔伯特变换-正弦波
- `TA_HT_TRENDLINE(data)` - TA-Lib希尔伯特变换-趋势线
- `TA_HT_TRENDMODE(data)` - TA-Lib希尔伯特变换-趋势vs周期模式
- `TA_KAMA(data, n)` - TA-Lib考夫曼自适应移动平均
- `TA_MAMA(data, fastlimit, slowlimit)` - TA-Lib MESA自适应移动平均
- `TA_MAVP(data, periods, minperiod, maxperiod, matype)` - TA-Lib可变周期移动平均
- `TA_MIDPOINT(data, n)` - TA-Lib中点价格
- `TA_MIDPRICE(data, n)` - TA-Lib中间价格
- `TA_SAR(data, acceleration, maximum)` - TA-Lib抛物线SAR
- `TA_SAREXT(data, startvalue, offsetonreverse, accelerationinitlong, accelerationlong, accelerationmaxlong, accelerationinitshort, accelerationshort, accelerationmaxshort)` - TA-Lib扩展抛物线SAR
- `TA_SMA(data, n)` - TA-Lib简单移动平均
- `TA_T3(data, n, vfactor)` - TA-Lib三重指数移动平均
- `TA_TEMA(data, n)` - TA-Lib三重指数移动平均
- `TA_TRIMA(data, n)` - TA-Lib三角移动平均
- `TA_WMA(data, n)` - TA-Lib加权移动平均

## 交易系统组件

### 信号指示器 (Signal)

信号指示器负责产生买入和卖出信号。

#### 工厂函数
- `SG_Flex(ind, slow_n=22)` - 基于EMA交叉的灵活信号指示器
- `SG_Cross(fast_ind, slow_ind)` - 双线交叉信号指示器
- `SG_Bool(buy_condition, sell_condition)` - 基于布尔条件的信号指示器
- `SG_Single(ind, filter_n=10, filter_p=0.1)` - 单线拐点信号指示器
- `SG_Single2(ind, filter_n=10, filter_p=0.1)` - 单线拐点信号指示器2

### 资金管理 (MoneyManager)

资金管理策略用于控制交易风险并决定每次买卖的数量。

#### 工厂函数
- `MM_FixedCount(num)` - 固定数量买入策略
- `MM_FixedPercent(percent)` - 固定比例资金投入策略
- `MM_FixedCapitalFunds(capital)` - 固定资金金额买入策略
- `MM_FixedRisk(p=0.02, max_loss=1000.0)` - 固定风险资金管理策略
- `MM_FixedUnits(unit)` - 固定单位买入策略
- `MM_WilliamsR(p=0.1, max_risk=1000.0)` - 威廉斯%R资金管理策略
- `MM_MaxRisk(max_risk=1000.0)` - 最大风险资金管理策略

### 止损策略 (Stoploss)

止损策略在交易亏损时生效。

#### 工厂函数
- `ST_FixedPercent(p=0.03)` - 固定百分比止损
- `ST_Atr(n=14, multiplier=3.0)` - 基于ATR的止损
- `ST_FixedValue(value=1000.0)` - 固定金额止损
- `ST_Indicator(ind, kpart="CLOSE")` - 基于指标的止损

### 止盈策略 (TakeProfit)

**重要说明**: hikyuu框架中的止盈策略实际上使用的是 `StoplossBase` 类，与止损策略共享相同的基类。在创建交易系统时，止盈策略通过 `tp` 参数设置。

#### 工厂函数
- `ST_FixedPercent(p=0.2)` - 固定百分比止盈（注意：使用ST_而非TP_前缀）
- `ST_FixedValue(value=1000.0)` - 固定金额止盈
- `ST_Indicator(ind, kpart="CLOSE")` - 基于指标的止盈

**API修正说明**: 
- ❌ 错误用法: `TP_FixedPercent(0.15)`
- ✅ 正确用法: `ST_FixedPercent(0.15)`

### 盈利目标策略 (ProfitGoal)

盈利目标策略是一种特殊的止盈策略。

#### 工厂函数
- `PG_FixedPercent(p=0.2)` - 固定百分比盈利目标
- `PG_FixedValue(value=1000.0)` - 固定金额盈利目标

### 环境判断策略 (Environment)

市场环境判断策略用于判断市场环境是否有效。

#### 工厂函数
- `EV_Bool(ind, value=0)` - 基于布尔值的环境判断
- `EV_TwoLine(ind1, ind2)` - 基于双线的环境判断

### 条件判断策略 (Condition)

系统有效条件用于判断系统本身的适用条件。

#### 工厂函数
- `CN_Bool(ind, value=0)` - 基于布尔值的条件判断
- `CN_OPLine(op=">=")` - 基于价格线的条件判断

### 移滑价差策略 (Slippage)

移滑价差算法仅用于回测，模拟实际交易中的价格差异。

#### 工厂函数
- `SP_FixedPercent(p=0.001)` - 固定百分比移滑价差
- `SP_FixedValue(value=0.01)` - 固定数值移滑价差

## 投资组合组件

### 选择器 (Selector)

选择器提供系统选择算法的接口，用于评估和选取交易系统。

#### 工厂函数
- `SE_Fixed(stock_list, sys)` - 固定选择器
- `SE_Signal(stock_list, sys)` - 基于信号的选择器
- `SE_PerformanceOptimal(stock_list, sys, mode="AR%")` - 绩效最优选择器

### 资产分配 (AllocateFunds)

资产分配算法用于在选定的系统中进行资产分配。

#### 工厂函数
- `AF_EqualWeight()` - 等权重分配
- `AF_FixedWeight(weight_list)` - 固定权重分配
- `AF_FixedWeightList(weight_list)` - 按权重列表分配

## 工厂函数

### 交易管理创建函数

- `crtTM(date=Datetime(), init_cash=100000, cost_func=TC_Zero(), name="SYS")` - 创建交易管理模块实例

### 成本算法函数

- `TC_TestStub()` - 测试存根成本算法
- `TC_Zero()` - 零成本算法
- `TC_FixedA(commission=5.0)` - 固定佣金成本算法

### 系统创建函数

- `SYS_Simple(tm=None, mm=None, ev=None, cn=None, sg=None, st=None, tp=None, pg=None, sp=None)` - 创建简单交易系统

**API修正说明**: SYS_Simple函数不接受 `name` 参数
- `SYS_WalkForward(sys_list, tm, train_len, test_len, se=None, train_tm=None)` - 创建滚动寻优系统

### 投资组合创建函数

- `PF_Simple(tm, se, af, adjust_cycle=1, adjust_mode="day", delay_to_trading_day=False)` - 创建简单投资组合
- `PF_WithoutAF(tm, se, adjust_cycle=1, adjust_mode="day", delay_to_trading_day=False, trade_on_close=False, sys_use_self_tm=False, sell_at_not_selected=True)` - 创建无资金分配算法的投资组合

## 工具函数

### 框架初始化

- `load_hikyuu(**kwargs)` - 初始化hikyuu框架
- `hikyuu_init(config_file, ignore_preload=False)` - 使用配置文件初始化

### 数据获取工具

- `getKData(market_code, query)` - 获取K线数据
- `getStock(market_code)` - 获取股票实例

### 选股工具

- `select(condition)` - 根据条件选股
- `select_by_rank(rank_ind, block, stock_list, n=10, asc=True)` - 根据排名选股

### 回测工具

- `backtest(on_bar, tm, start_date, end_date, ktype=Query.DAY)` - 事件驱动回测
- `run_in_strategy(pf, ktype, obs, tc)` - 在策略中运行投资组合

### 数据处理工具

- `to_np(data)` - 转换为numpy数组
- `to_df(data)` - 转换为pandas DataFrame

### 实时数据工具

- `startSpotAgent(worker_num=1, addr="ipc:///tmp/hikyuu_real.ipc")` - 启动实时数据代理
- `stopSpotAgent()` - 停止实时数据代理
- `getGlobalSpotAgent()` - 获取全局实时数据代理

### 绘图工具

- `matplotlib_draw.iplot(data, new=True, axes=None, legend_on=False)` - 绘制指标
- `matplotlib_draw.ibar(data, new=True, axes=None, legend_on=False)` - 绘制柱状图
- `matplotlib_draw.kplot(kdata, new=True, axes=None, colorup='r', colordown='g')` - 绘制K线图

### 其他工具函数

- `roundEx(val, ndigits=2)` - 扩展的四舍五入函数
- `roundUp(val, ndigits=2)` - 向上取整
- `roundDown(val, ndigits=2)` - 向下取整

## 枚举和常量

### K线类型枚举 (KType)

```python
Query.INVALID_KTYPE    # 无效K线类型
Query.DAY              # 日线
Query.WEEK             # 周线  
Query.MONTH            # 月线
Query.QUARTER          # 季线
Query.HALFYEAR         # 半年线
Query.YEAR             # 年线
Query.MIN              # 1分钟线
Query.MIN5             # 5分钟线
Query.MIN15            # 15分钟线
Query.MIN30            # 30分钟线
Query.MIN60            # 60分钟线
Query.HOUR2            # 2小时线
Query.HOUR4            # 4小时线
Query.HOUR6            # 6小时线
Query.HOUR12           # 12小时线
```

### 复权类型枚举 (RecoverType)

```python
Query.NO_RECOVER       # 不复权
Query.FORWARD          # 前复权
Query.BACKWARD         # 后复权
Query.EQUAL_FORWARD    # 等比前复权
Query.EQUAL_BACKWARD   # 等比后复权
```

### 查询类型枚举 (QueryType)

```python
Query.INDEX            # 按索引查询
Query.DATE             # 按日期查询
```

### 交易业务类型 (Business)

```python
Business.INIT          # 初始化
Business.BUY           # 买入
Business.SELL          # 卖出
Business.GIFT          # 送股
Business.BONUS         # 分红
Business.CHECKIN       # 存入资金
Business.CHECKOUT      # 取出资金
Business.CHECKIN_STOCK # 存入股票
Business.CHECKOUT_STOCK # 取出股票
Business.BORROW_CASH   # 融资
Business.RETURN_CASH   # 归还融资
Business.BORROW_STOCK  # 融券
Business.RETURN_STOCK  # 归还融券
Business.SELL_SHORT    # 卖空
Business.BUY_SHORT     # 买空
```

### 全局常量

```python
# 行情数据指标常量（在交互模式下可用）
O = OPEN()      # 开盘价
H = HIGH()      # 最高价
L = LOW()       # 最低价
C = CLOSE()     # 收盘价
A = AMO()       # 成交金额
V = VOL()       # 成交量

# 全局StockManager实例
sm = StockManager.instance()
```

### 参数常量

```python
# 系统参数
"buy_delay"           # 买入延迟
"sell_delay"          # 卖出延迟
"max_delay_count"     # 最大延迟次数
"delay_use_current_price"  # 延迟是否使用当前价格
"precision"           # 价格精度
"support_borrow_cash" # 是否支持融资
"support_borrow_stock" # 是否支持融券
"save_action"         # 是否保存交易动作
```

---

**使用说明**:

1. **导入框架**: `from hikyuu import *`
2. **初始化**: `load_hikyuu()`  
3. **获取数据**: `sm['sh000001'].get_kdata(Query(-100))`
4. **技术指标**: `MA(CLOSE(), 20)`
5. **交易系统**: `SYS_Simple(tm=crtTM(), sg=SG_Cross(MA(CLOSE(), 5), MA(CLOSE(), 20)), mm=MM_FixedPercent(0.1))`
6. **投资组合**: `PF_Simple(tm=crtTM(), se=SE_Fixed(stocks, sys), af=AF_EqualWeight())`
7. **绩效分析**: `tm.get_performance(Datetime.now(), Query.DAY)`

**API修正示例**:

```python
# 正确的交易系统创建方式
tm = crtTM(init_cash=1000000, name="策略名称")
sg = SG_Cross(MA(CLOSE(), 5), MA(CLOSE(), 20))
mm = MM_FixedPercent(0.2)
st = ST_FixedPercent(0.05)  # 止损
tp = ST_FixedPercent(0.15)  # 止盈（注意使用ST_而非TP_）

# 创建系统（注意：不要传入name参数）
sys = SYS_Simple(tm=tm, sg=sg, mm=mm, st=st, tp=tp)

# MACD指标的正确访问方式
C = CLOSE()
C.set_context(stock, query)
macd = MACD(C, 12, 26, 9)
if len(macd) > 0:
    dif = macd.get_result(0)      # DIF线
    dea = macd.get_result(1)      # DEA线  
    histogram = macd.get_result(2) # 柱状图
```

**注意事项**:

- 所有指标都需要设置上下文才能计算结果
- 在事件驱动回测中避免使用未来数据
- 合理设置预加载参数以优化性能
- 定期保存和备份重要数据

**常见API错误及修正**:

1. **止盈策略API错误**:
   - ❌ `TP_FixedPercent(0.15)` 
   - ✅ `ST_FixedPercent(0.15)`

2. **SYS_Simple参数错误**:
   - ❌ `SYS_Simple(tm=tm, sg=sg, mm=mm, name="系统名")`
   - ✅ `SYS_Simple(tm=tm, sg=sg, mm=mm)`

3. **MACD指标访问错误**:
   - ❌ `dif = macd[0]`
   - ✅ `dif = macd.get_result(0)`

4. **指标值访问错误**:
   - ❌ `print(f"MA值: {ma[-1]:.2f}")`
   - ✅ `if len(ma) > 0: print(f"MA值: {ma[-1]:.2f}")`

5. **KData时间访问**:
   - ✅ `kdata[0].datetime` 和 `kdata[-1].datetime`（通过KRecord对象访问）

---

**文档编制**: 基于DeepWiki对fasiondog/hikyuu项目的完整API分析  
**最后更新**: 2025年8月25日
