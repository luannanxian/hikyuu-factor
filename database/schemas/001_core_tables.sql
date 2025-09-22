-- Hikyuu Factor Database Schema
-- 核心表结构定义

-- 设置字符集和时区
SET NAMES utf8mb4;
SET TIME_ZONE = '+08:00';

-- =======================================================================================
-- 基础数据表
-- =======================================================================================

-- 股票基础信息表
CREATE TABLE IF NOT EXISTS stocks (
    stock_code VARCHAR(20) PRIMARY KEY COMMENT '股票代码 (如: sh000001)',
    stock_name VARCHAR(100) NOT NULL COMMENT '股票名称',
    market VARCHAR(10) NOT NULL COMMENT '市场代码 (sh/sz/bj)',
    sector VARCHAR(50) COMMENT '行业板块',
    list_date DATE COMMENT '上市日期',
    delist_date DATE COMMENT '退市日期',
    status ENUM('active', 'suspended', 'delisted') DEFAULT 'active' COMMENT '状态',

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    INDEX idx_market (market),
    INDEX idx_sector (sector),
    INDEX idx_status (status),
    INDEX idx_list_date (list_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='股票基础信息表';

-- 市场数据表 (日K线数据)
CREATE TABLE IF NOT EXISTS market_data (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    stock_code VARCHAR(20) NOT NULL COMMENT '股票代码',
    trade_date DATE NOT NULL COMMENT '交易日期',

    open_price DECIMAL(10,3) NOT NULL COMMENT '开盘价',
    high_price DECIMAL(10,3) NOT NULL COMMENT '最高价',
    low_price DECIMAL(10,3) NOT NULL COMMENT '最低价',
    close_price DECIMAL(10,3) NOT NULL COMMENT '收盘价',
    volume BIGINT NOT NULL COMMENT '成交量',
    amount DECIMAL(18,2) NOT NULL COMMENT '成交金额',

    adj_factor DECIMAL(10,6) DEFAULT 1.0 COMMENT '复权因子',
    turnover_rate DECIMAL(8,4) COMMENT '换手率',

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    UNIQUE KEY uk_stock_date (stock_code, trade_date),
    INDEX idx_trade_date (trade_date),
    INDEX idx_stock_code (stock_code),

    FOREIGN KEY (stock_code) REFERENCES stocks(stock_code) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='市场数据表'
PARTITION BY RANGE COLUMNS(trade_date) (
    PARTITION p2020 VALUES LESS THAN ('2021-01-01'),
    PARTITION p2021 VALUES LESS THAN ('2022-01-01'),
    PARTITION p2022 VALUES LESS THAN ('2023-01-01'),
    PARTITION p2023 VALUES LESS THAN ('2024-01-01'),
    PARTITION p2024 VALUES LESS THAN ('2025-01-01'),
    PARTITION p_future VALUES LESS THAN MAXVALUE
);

-- 财务数据表
CREATE TABLE IF NOT EXISTS financial_data (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    stock_code VARCHAR(20) NOT NULL COMMENT '股票代码',
    report_date DATE NOT NULL COMMENT '报告期',

    -- 基本财务指标
    pe_ratio DECIMAL(10,3) COMMENT '市盈率',
    pb_ratio DECIMAL(10,3) COMMENT '市净率',
    roe DECIMAL(8,4) COMMENT '净资产收益率',
    roa DECIMAL(8,4) COMMENT '总资产收益率',

    -- 财务数据
    total_revenue DECIMAL(18,2) COMMENT '营业收入',
    net_profit DECIMAL(18,2) COMMENT '净利润',
    total_assets DECIMAL(18,2) COMMENT '总资产',
    total_equity DECIMAL(18,2) COMMENT '股东权益',

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    UNIQUE KEY uk_stock_report (stock_code, report_date),
    INDEX idx_report_date (report_date),

    FOREIGN KEY (stock_code) REFERENCES stocks(stock_code) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='财务数据表';

-- =======================================================================================
-- 因子相关表
-- =======================================================================================

-- 因子定义表
CREATE TABLE IF NOT EXISTS factor_definitions (
    factor_id VARCHAR(50) PRIMARY KEY COMMENT '因子ID',
    factor_name VARCHAR(100) NOT NULL COMMENT '因子名称',
    category ENUM('momentum', 'value', 'quality', 'volatility', 'growth', 'technical') NOT NULL COMMENT '因子类别',

    formula TEXT COMMENT 'Hikyuu计算公式',
    description TEXT COMMENT '因子描述',
    economic_logic TEXT COMMENT '经济学逻辑',

    status ENUM('active', 'inactive', 'testing') DEFAULT 'active' COMMENT '状态',

    -- 计算参数
    calculation_params JSON COMMENT '计算参数 JSON格式',

    -- 统计信息
    coverage_ratio DECIMAL(5,4) DEFAULT 0 COMMENT '覆盖率',
    avg_calculation_time_ms INT DEFAULT 0 COMMENT '平均计算时间(毫秒)',

    created_by VARCHAR(50) COMMENT '创建者',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    INDEX idx_category (category),
    INDEX idx_status (status),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='因子定义表';

-- 因子值表 (核心表，数据量大)
CREATE TABLE IF NOT EXISTS factor_values (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    factor_id VARCHAR(50) NOT NULL COMMENT '因子ID',
    stock_code VARCHAR(20) NOT NULL COMMENT '股票代码',
    trade_date DATE NOT NULL COMMENT '交易日期',

    factor_value DECIMAL(15,6) NOT NULL COMMENT '因子原始值',
    factor_score DECIMAL(8,6) COMMENT '因子标准化分数 (0-1)',
    percentile_rank DECIMAL(5,4) COMMENT '百分位排名',

    -- 计算元信息
    calculation_id VARCHAR(50) COMMENT '计算批次ID',
    data_version VARCHAR(20) COMMENT '数据版本',

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE KEY uk_factor_stock_date (factor_id, stock_code, trade_date),
    INDEX idx_factor_date (factor_id, trade_date),
    INDEX idx_stock_date (stock_code, trade_date),
    INDEX idx_calculation_id (calculation_id),

    FOREIGN KEY (factor_id) REFERENCES factor_definitions(factor_id) ON DELETE CASCADE,
    FOREIGN KEY (stock_code) REFERENCES stocks(stock_code) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='因子值表'
PARTITION BY RANGE COLUMNS(trade_date) (
    PARTITION p2020 VALUES LESS THAN ('2021-01-01'),
    PARTITION p2021 VALUES LESS THAN ('2022-01-01'),
    PARTITION p2022 VALUES LESS THAN ('2023-01-01'),
    PARTITION p2023 VALUES LESS THAN ('2024-01-01'),
    PARTITION p2024 VALUES LESS THAN ('2025-01-01'),
    PARTITION p_future VALUES LESS THAN MAXVALUE
);

-- 因子计算任务表
CREATE TABLE IF NOT EXISTS factor_calculation_tasks (
    task_id VARCHAR(50) PRIMARY KEY COMMENT '任务ID',
    factor_ids JSON NOT NULL COMMENT '因子ID列表',
    stock_list JSON COMMENT '股票列表 (null表示全市场)',

    date_range JSON NOT NULL COMMENT '日期范围 {start_date, end_date}',

    status ENUM('pending', 'running', 'completed', 'failed', 'cancelled') DEFAULT 'pending' COMMENT '状态',
    progress DECIMAL(5,2) DEFAULT 0 COMMENT '进度百分比',

    -- 执行信息
    started_at TIMESTAMP NULL COMMENT '开始时间',
    completed_at TIMESTAMP NULL COMMENT '完成时间',
    error_message TEXT COMMENT '错误信息',

    -- 统计信息
    total_calculations INT DEFAULT 0 COMMENT '总计算次数',
    successful_calculations INT DEFAULT 0 COMMENT '成功计算次数',

    created_by VARCHAR(50) COMMENT '创建者',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    INDEX idx_status (status),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='因子计算任务表';