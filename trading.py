import os
import ta
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, time, timedelta
from pykis import KisAuth, KisClient, KisStock
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class PairTradingBacktest:
    def __init__(self, strategy_params=None):
        self.strategy_params = strategy_params or {
            'profit_target': 0.01,
            'stop_loss': 0.007,
            'max_holding_time': 180,
            'volume_threshold': 2.5,
            'rsi_oversold': 35,
            'rsi_overbought': 65
        }
        self.trades = []
        self.positions = {'TSLL': None, 'TSLQ': None}
        self.entry_prices = {'TSLL': 0, 'TSLQ': 0}
        self.entry_times = {'TSLL': None, 'TSLQ': None}
        
    def run_backtest(self, df_long, df_short, start_date=None, end_date=None):
        """백테스트 실행"""
        results = {
            'trades': [],
            'profit_history': [],
            'metrics': {}
        }
        
        # 데이터 필터링
        if start_date:
            df_long = df_long[df_long['timestamp'] >= start_date]
            df_short = df_short[df_short['timestamp'] >= start_date]
        if end_date:
            df_long = df_long[df_long['timestamp'] <= end_date]
            df_short = df_short[df_short['timestamp'] <= end_date]
            
        # 백테스트 실행
        for i in range(len(df_long)):
            current_time = df_long.iloc[i]['timestamp']
            current_prices = {
                'TSLL': df_long.iloc[i]['price'],
                'TSLQ': df_short.iloc[i]['price']
            }
            
            # 포지션 없을 때 진입 조건 확인
            if not self.positions['TSLL']:
                if self.check_entry_conditions(df_long.iloc[:i+1], df_short.iloc[:i+1]):
                    self.open_position(current_time, current_prices)
                    
            # 포지션 있을 때 청산 조건 확인
            elif self.check_exit_conditions(current_time, current_prices):
                trade_result = self.close_position(current_time, current_prices)
                results['trades'].append(trade_result)
                results['profit_history'].append(trade_result['total_profit'])
                
        # 성과 지표 계산
        results['metrics'] = self.calculate_performance_metrics(results['trades'])
        return results
    
    def check_entry_conditions(self, df_long, df_short):
        """진입 조건 확인"""
        if len(df_long) < 20:
            return False
            
        long_data = df_long.iloc[-1]
        short_data = df_short.iloc[-1]
        
        return (
            long_data['rsi'] < self.strategy_params['rsi_oversold'] and
            short_data['rsi'] > self.strategy_params['rsi_overbought'] and
            long_data['volume_ratio'] > self.strategy_params['volume_threshold'] and
            short_data['volume_ratio'] > self.strategy_params['volume_threshold'] and
            long_data['price'] < long_data['bb_lower'] and
            short_data['price'] > short_data['bb_upper']
        )
    
    def check_exit_conditions(self, current_time, current_prices):
        """청산 조건 확인"""
        if not self.positions['TSLL']:
            return False
            
        holding_time = (current_time - self.entry_times['TSLL']).total_seconds()
        
        profit_long = (current_prices['TSLL'] - self.entry_prices['TSLL']) / self.entry_prices['TSLL']
        profit_short = (self.entry_prices['TSLQ'] - current_prices['TSLQ']) / self.entry_prices['TSLQ']
        total_profit = profit_long + profit_short
        
        return (
            total_profit >= self.strategy_params['profit_target'] or
            total_profit <= -self.strategy_params['stop_loss'] or
            holding_time >= self.strategy_params['max_holding_time']
        )
    
    def open_position(self, time, prices):
        """포지션 진입"""
        self.positions = {'TSLL': 'long', 'TSLQ': 'short'}
        self.entry_prices = prices.copy()
        self.entry_times = {'TSLL': time, 'TSLQ': time}
    
    def close_position(self, time, prices):
        """포지션 청산"""
        profit_long = (prices['TSLL'] - self.entry_prices['TSLL']) / self.entry_prices['TSLL']
        profit_short = (self.entry_prices['TSLQ'] - prices['TSLQ']) / self.entry_prices['TSLQ']
        total_profit = profit_long + profit_short
        
        trade_result = {
            'entry_time': self.entry_times['TSLL'],
            'exit_time': time,
            'holding_period': (time - self.entry_times['TSLL']).total_seconds() / 60,
            'entry_prices': self.entry_prices.copy(),
            'exit_prices': prices.copy(),
            'profit_long': profit_long,
            'profit_short': profit_short,
            'total_profit': total_profit
        }
        
        self.positions = {'TSLL': None, 'TSLQ': None}
        return trade_result
    
    def calculate_performance_metrics(self, trades):
        """성과 지표 계산"""
        if not trades:
            return {}
            
        profits = [trade['total_profit'] for trade in trades]
        
        return {
            'total_trades': len(trades),
            'winning_trades': sum(1 for p in profits if p > 0),
            'losing_trades': sum(1 for p in profits if p <= 0),
            'win_rate': sum(1 for p in profits if p > 0) / len(profits),
            'total_return': sum(profits),
            'avg_return': np.mean(profits),
            'max_return': max(profits),
            'min_return': min(profits),
            'avg_holding_period': np.mean([trade['holding_period'] for trade in trades]),
            'sharpe_ratio': np.mean(profits) / np.std(profits) if len(profits) > 1 else 0,
            'max_drawdown': self.calculate_max_drawdown(profits)
        }
    
    def calculate_max_drawdown(self, profits):
        """최대 손실폭 계산"""
        cumulative = np.cumsum(profits)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        return np.max(drawdown) if len(drawdown) > 0 else 0

def run_strategy_optimization(backtest, df_long, df_short, param_ranges):
    """전략 최적화"""
    best_result = None
    best_params = None
    best_sharpe = float('-inf')
    
    for params in generate_param_combinations(param_ranges):
        backtest.strategy_params = params
        result = backtest.run_backtest(df_long, df_short)
        
        if result['metrics'].get('sharpe_ratio', float('-inf')) > best_sharpe:
            best_sharpe = result['metrics']['sharpe_ratio']
            best_result = result
            best_params = params
            
    return best_params, best_result

def generate_param_combinations(param_ranges):
    """파라미터 조합 생성"""
    import itertools
    
    keys = param_ranges.keys()
    values = param_ranges.values()
    combinations = itertools.product(*values)
    
    return [dict(zip(keys, combo)) for combo in combinations]

class LeveragePairTradingSystem:
    def __init__(self):
        """
        레버리지 페어 트레이딩 시스템 초기화
        """
        # PyKIS 초기화
        load_dotenv()
        try:
            self.kis = KisClient("secret.json", token="token.json", keep_token=True)
            print("저장된 토큰을 사용하여 초기화 성공")
        except Exception as e:
            print(f"토큰 초기화 실패, 새로운 인증 시도: {str(e)}")
            auth = KisAuth(
                id=os.getenv('KIS_ID'),
                appkey=os.getenv('KIS_APP_KEY'),
                secretkey=os.getenv('KIS_SECRET'),
                account=os.getenv('KIS_ACCOUNT'),
                virtual=False,
                keep_token=True, 
            )
            auth.save("secret.json")
            self.kis = KisClient("secret.json", keep_token=True)
            self.kis.token.save("token.json")

        # 포지션 정보
        self.positions = {
            'TSLL': None,  # 롱 레버리지
            'TSLQ': None   # 숏 레버리지
        }
        self.entry_prices = {'TSLL': 0, 'TSLQ': 0}
        self.entry_times = {'TSLL': None, 'TSLQ': None}
        
        # 데이터 저장
        self.price_data = {'TSLL': pd.DataFrame(), 'TSLQ': pd.DataFrame()}
        self.last_update = datetime.now()
        
        # 예측 모델
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = MinMaxScaler()
        self.is_model_trained = False

        # 리스크 관리 파라미터 초기화
        self.update_risk_parameters()

    

    def update_risk_parameters(self):
        """
        거래 시간대별 리스크 파라미터 설정
        """
        trading_type = self.check_market_time()
        
        if trading_type == "REGULAR":
            self.profit_target = 0.01    # 1.0%
            self.stop_loss = 0.007       # 0.7%
            self.max_holding_time = 180   # 3분
            self.volume_threshold = 2.5    # 거래량 임계값
            self.rsi_oversold = 35        # RSI 과매도
            self.rsi_overbought = 65      # RSI 과매수
        else:  # 시간외 거래
            self.profit_target = 0.012    # 1.2%
            self.stop_loss = 0.005        # 0.5%
            self.max_holding_time = 300    # 5분
            self.volume_threshold = 3.0    # 더 높은 거래량 요구
            self.rsi_oversold = 30        # 더 엄격한 RSI
            self.rsi_overbought = 70

    def check_market_time(self):
        """
        현재 거래 시간 확인
        """
        current_time = datetime.now().time()
        current_weekday = datetime.now().weekday()
        
        if current_weekday >= 5:
            return "CLOSED"
        
        market_times = {
            'pre_market_start': time(18, 0),
            'pre_market_end': time(23, 30),
            'regular_end': time(4, 0),
            'after_market_start': time(5, 0),
            'after_market_end': time(9, 0)
        }
        
        if market_times['pre_market_start'] <= current_time <= market_times['pre_market_end']:
            return "PRE_MARKET"
        elif market_times['pre_market_end'] <= current_time or current_time <= market_times['regular_end']:
            return "REGULAR"
        elif market_times['after_market_start'] <= current_time <= market_times['after_market_end']:
            return "AFTER_MARKET"
        
        return "CLOSED"

    def calculate_indicators(self, df):
        """
        기술적 지표 계산
        """
        if len(df) < 20:
            return None
            
        # 볼린저 밴드
        bb = ta.volatility.BollingerBands(df['price'], window=20)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['price'], window=9).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['price'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        
        # 거래량 지표
        df['volume_ma'] = df['volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        return df

    def train_prediction_model(self, df):
        """
        예측 모델 학습
        """
        try:
            # 사용 가능한 특성 확인
            print("\n사용 가능한 컬럼:", df.columns.tolist())
            
            feature_columns = [
                'rsi', 'macd', 'macd_signal', 'volume_ratio',
                'bb_upper', 'bb_lower', 'bb_middle'  # ma5, ma10 대신 다른 지표 사용
            ]
            
            # 필요한 특성이 모두 있는지 확인
            missing_features = [col for col in feature_columns if col not in df.columns]
            if missing_features:
                raise ValueError(f"누락된 특성: {missing_features}")
            
            X = df[feature_columns].fillna(0)
            y = df['price'].shift(-1)  # 다음 가격을 예측
            
            # NaN 제거
            X = X[:-1]
            y = y[:-1].fillna(0)
            
            # 데이터 스케일링
            X_scaled = self.scaler.fit_transform(X)
            
            # 모델 학습
            self.model.fit(X_scaled, y)
            self.is_model_trained = True
            
            print("예측 모델 학습이 완료되었습니다.")
            return True
            
        except Exception as e:
            print(f"모델 학습 중 에러 발생: {str(e)}")
            return False

    def check_entry_conditions(self, df_long, df_short):
        """
        진입 조건 확인
        """
        if len(df_long) < 20 or len(df_short) < 20:
            return False

        # TSLL 조건
        long_rsi = df_long['rsi'].iloc[-1]
        long_volume_ratio = df_long['volume_ratio'].iloc[-1]
        long_price = df_long['price'].iloc[-1]
        long_bb_lower = df_long['bb_lower'].iloc[-1]

        # TSLQ 조건
        short_rsi = df_short['rsi'].iloc[-1]
        short_volume_ratio = df_short['volume_ratio'].iloc[-1]
        short_price = df_short['price'].iloc[-1]
        short_bb_upper = df_short['bb_upper'].iloc[-1]

        # 진입 조건
        if (long_rsi < self.rsi_oversold and 
            short_rsi > self.rsi_overbought and
            long_volume_ratio > self.volume_threshold and
            short_volume_ratio > self.volume_threshold and
            long_price < long_bb_lower and
            short_price > short_bb_upper):
            return True

        return False

    def execute_paired_trade(self, stock_long, stock_short, action, price_long=None, price_short=None, quantity=1):
        """
        페어 트레이딩 실행
        """
        try:
            current_time = datetime.now()
            
            if action == "entry":
                # TSLL 매수
                long_order = stock_long.buy(qty=quantity)
                print(f"\n[TSLL 매수] 가격: {price_long:,.2f}")
                
                # TSLQ 매도
                short_order = stock_short.sell(qty=quantity)
                print(f"[TSLQ 매도] 가격: {price_short:,.2f}")
                
                self.positions['TSLL'] = 'long'
                self.positions['TSLQ'] = 'short'
                self.entry_prices['TSLL'] = price_long
                self.entry_prices['TSLQ'] = price_short
                self.entry_times['TSLL'] = current_time
                self.entry_times['TSLQ'] = current_time
                
            elif action == "exit":
                # 수익률 계산
                profit_long = (price_long - self.entry_prices['TSLL']) / self.entry_prices['TSLL']
                profit_short = (self.entry_prices['TSLQ'] - price_short) / self.entry_prices['TSLQ']
                total_profit = profit_long + profit_short
                
                # 포지션 청산
                stock_long.sell(qty=quantity)
                stock_short.buy(qty=quantity)
                
                print(f"\n[청산 완료] {current_time.strftime('%H:%M:%S')}")
                print(f"TSLL 수익률: {profit_long*100:.2f}%")
                print(f"TSLQ 수익률: {profit_short*100:.2f}%")
                print(f"전체 수익률: {total_profit*100:.2f}%")
                
                self.positions['TSLL'] = None
                self.positions['TSLQ'] = None
                
            return True
            
        except Exception as e:
            print(f"주문 실행 중 에러: {str(e)}")
            return False

    def check_exit_conditions(self, current_prices):
        """
        청산 조건 확인
        """
        if not self.positions['TSLL'] or not self.positions['TSLQ']:
            return False

        current_time = datetime.now()
        holding_time = (current_time - self.entry_times['TSLL']).total_seconds()

        # 수익률 계산
        profit_long = (current_prices['TSLL'] - self.entry_prices['TSLL']) / self.entry_prices['TSLL']
        profit_short = (self.entry_prices['TSLQ'] - current_prices['TSLQ']) / self.entry_prices['TSLQ']
        total_profit = profit_long + profit_short

        # 청산 조건
        if (total_profit >= self.profit_target or
            total_profit <= -self.stop_loss or
            holding_time >= self.max_holding_time):
            return True

        return False

    def on_price_update(self, symbol, price, volume):
        """
        가격 업데이트 처리 및 상태 출력
        """
        try:
            current_time = datetime.now()
            
            # 데이터 저장
            new_data = {
                'timestamp': current_time,
                'price': price,
                'volume': volume
            }
            
            new_df = pd.DataFrame([new_data])
            if self.price_data[symbol].empty:
                self.price_data[symbol] = new_df
            else:
                self.price_data[symbol] = pd.concat([self.price_data[symbol], new_df], ignore_index=True)
            
            # 20분 이전 데이터 제거
            self.price_data[symbol] = self.price_data[symbol][
                self.price_data[symbol]['timestamp'] > current_time - timedelta(minutes=20)
            ]

            # 1분마다 기본 상태 출력
            if not hasattr(self, 'last_status_print'):
                self.last_status_print = current_time
            
            if (current_time - self.last_status_print).total_seconds() >= 60:
                print(f"\n[{current_time.strftime('%H:%M:%S')} - 시스템 상태]")
                print(f"TSLL 현재가: {self.price_data['TSLL']['price'].iloc[-1]:,.2f}" if 'TSLL' in self.price_data and not self.price_data['TSLL'].empty else "TSLL 데이터 없음")
                print(f"TSLQ 현재가: {self.price_data['TSLQ']['price'].iloc[-1]:,.2f}" if 'TSLQ' in self.price_data and not self.price_data['TSLQ'].empty else "TSLQ 데이터 없음")
                
                # 포지션 보유 중일 때 수익률 표시
                if self.positions['TSLL'] and self.positions['TSLQ']:
                    current_prices = {
                        'TSLL': self.price_data['TSLL']['price'].iloc[-1],
                        'TSLQ': self.price_data['TSLQ']['price'].iloc[-1]
                    }
                    profit_long = (current_prices['TSLL'] - self.entry_prices['TSLL']) / self.entry_prices['TSLL'] * 100
                    profit_short = (self.entry_prices['TSLQ'] - current_prices['TSLQ']) / self.entry_prices['TSLQ'] * 100
                    total_profit = profit_long + profit_short
                    print(f"현재 총 수익률: {total_profit:.2f}%")
                
                self.last_status_print = current_time

            # 지표 계산
            df_long = self.calculate_indicators(self.price_data['TSLL'])
            df_short = self.calculate_indicators(self.price_data['TSLQ'])
            
            if df_long is not None and df_short is not None:
                # 예측 모델 학습 상태
                if len(self.price_data['TSLL']) >= 100 and not self.is_model_trained:
                    print("\n[시스템 알림] 예측 모델 학습 완료")
                    self.train_prediction_model(df_long)  # TSLL 데이터로 학습

                # 매매 신호 확인
                current_prices = {
                    'TSLL': df_long['price'].iloc[-1],
                    'TSLQ': df_short['price'].iloc[-1]
                }

                if not self.positions['TSLL'] and not self.positions['TSLQ']:
                    if self.check_entry_conditions(df_long, df_short):
                        print(f"\n[매수 신호 감지] {current_time.strftime('%H:%M:%S')}")
                        print(f"TSLL RSI: {df_long['rsi'].iloc[-1]:.2f}")
                        print(f"TSLQ RSI: {df_short['rsi'].iloc[-1]:.2f}")
                else:
                    if self.check_exit_conditions(current_prices):
                        profit_long = (current_prices['TSLL'] - self.entry_prices['TSLL']) / self.entry_prices['TSLL'] * 100
                        profit_short = (self.entry_prices['TSLQ'] - current_prices['TSLQ']) / self.entry_prices['TSLQ'] * 100
                        total_profit = profit_long + profit_short
                        print(f"\n[매도 신호 감지] {current_time.strftime('%H:%M:%S')}")
                        print(f"총 수익률: {total_profit:.2f}%")
                    
        except Exception as e:
            print(f"\n[오류 발생] {str(e)}")

    def run_backtest_analysis(self):
        """백테스트 분석 실행"""
        try:
            backtest = PairTradingBacktest()
            results = backtest.run_backtest(
                self.price_data['TSLL'],
                self.price_data['TSLQ']
            )
            
            print("\n[백테스트 결과]")
            print(f"총 거래 횟수: {results['metrics']['total_trades']}")
            print(f"승률: {results['metrics']['win_rate']*100:.2f}%")
            print(f"총 수익률: {results['metrics']['total_return']*100:.2f}%")
            print(f"평균 보유기간: {results['metrics']['avg_holding_period']:.1f}분")
            print(f"샤프 비율: {results['metrics']['sharpe_ratio']:.2f}")
            print(f"최대 손실폭: {results['metrics']['max_drawdown']*100:.2f}%")
            
            return results
            
        except Exception as e:
            print(f"백테스트 실행 중 오류: {str(e)}")
            return None    

    def start_trading(self):
        """
        트레이딩 시작
        """
        try:
            # 종목 설정
            stock_long = self.kis.stock("TSLL")
            stock_short = self.kis.stock("TSLQ")
            
            print(f"\n[페어 트레이딩 시작]")
            print(f"Long: TSLL (테슬라 롱 레버리지)")
            print(f"Short: TSLQ (테슬라 숏 레버리지)")
            print(f"거래시간: {self.check_market_time()}")
            
            def on_price_update_long(sender, e):
                self.on_price_update('TSLL', float(e.response.price), float(e.response.volume))
                if len(self.price_data['TSLL']) > 0 and len(self.price_data['TSLQ']) > 0:
                    self.process_trading_logic(stock_long, stock_short)
                
            def on_price_update_short(sender, e):
                self.on_price_update('TSLQ', float(e.response.price), float(e.response.volume))
            
            # 실시간 모니터링 시작
            ticket_long = stock_long.on("price", on_price_update_long)
            ticket_short = stock_short.on("price", on_price_update_short)
            
            return ticket_long, ticket_short
            
        except Exception as e:
            print(f"트레이딩 시작 중 에러: {str(e)}")
            return None, None

    def process_trading_logic(self, stock_long, stock_short):
        """
        트레이딩 로직 처리
        """
        try:
            # 지표 계산
            df_long = self.calculate_indicators(self.price_data['TSLL'])
            df_short = self.calculate_indicators(self.price_data['TSLQ'])
            
            if df_long is None or df_short is None:
                return
                
            current_prices = {
                'TSLL': df_long['price'].iloc[-1],
                'TSLQ': df_short['price'].iloc[-1]
            }
            
            # 포지션이 없을 때 진입 조건 확인
            if not self.positions['TSLL'] and not self.positions['TSLQ']:
                if self.check_entry_conditions(df_long, df_short):
                    self.execute_paired_trade(stock_long, stock_short, "entry",
                                           price_long=current_prices['TSLL'],
                                           price_short=current_prices['TSLQ'])
            
            # 포지션이 있을 때 청산 조건 확인
            elif self.check_exit_conditions(current_prices):
                self.execute_paired_trade(stock_long, stock_short, "exit",
                                       price_long=current_prices['TSLL'],
                                       price_short=current_prices['TSLQ'])
                
        except Exception as e:
            print(f"트레이딩 로직 처리 중 에러: {str(e)}")

def main():
    try:
        system = LeveragePairTradingSystem()
        ticket_long, ticket_short = system.start_trading()
        
        if ticket_long and ticket_short:
            print("\n시스템이 실행 중입니다. 종료하려면 Enter를 누르세요.")
            input()
            ticket_long.unsubscribe()
            ticket_short.unsubscribe()
            print("시스템이 종료되었습니다.")
            
    except Exception as e:
        print(f"시스템 실행 중 에러 발생: {str(e)}")

if __name__ == "__main__":
    main()
