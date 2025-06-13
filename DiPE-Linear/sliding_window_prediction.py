# fixed_sliding_window_prediction.py
"""
수정된 DiPE 모델 로드 방식
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yaml
import os
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class FixedSlidingWindowPredictor:
    def __init__(self, model_path, data_path):
        self.model_path = model_path
        self.data_path = data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델 로드
        self.model = self.load_model_fixed()
        
        # 데이터 로드
        self.data = self.load_data()
        
        # 스케일러 설정
        self.scaler = self.setup_scaler()
        
        print(f"✅ 수정된 Sliding Window 예측기 준비 완료")

    def load_model_fixed(self):
        """수정된 DiPE 모델 로드 (키 이름 문제 해결)"""
        try:
            print("🔄 모델 로드 시도...")
            
            # 체크포인트 로드
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            print("📋 체크포인트 키 확인...")
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print(f"   state_dict 키 개수: {len(state_dict)}")
                
                # 첫 몇 개 키 이름 확인
                sample_keys = list(state_dict.keys())[:5]
                print(f"   샘플 키: {sample_keys}")
                
            else:
                state_dict = checkpoint
                print("   직접 state_dict 사용")
            
            # DiPE 모델 생성
            from timeprophet.models.DiPE import DiPE
            
            model = DiPE(
                input_len=30,
                output_len=7,
                input_features=402,
                output_features=402,
                individual_f=True,
                individual_t=True,
                individual_c=False,
                num_experts=4,
                use_revin=True,
                use_time_w=True,
                use_freq_w=True,
                loss_alpha=0.7,
                t_loss='mae'
            )
            
            # 키 이름 수정
            print("🔧 키 이름 수정 중...")
            corrected_state_dict = {}
            
            for key, value in state_dict.items():
                # "model." 접두사 제거
                if key.startswith('model.'):
                    new_key = key[6:]  # "model." 제거
                    corrected_state_dict[new_key] = value
                    print(f"   수정: {key} → {new_key}")
                else:
                    corrected_state_dict[key] = value
            
            # 수정된 state_dict 로드
            missing_keys, unexpected_keys = model.load_state_dict(corrected_state_dict, strict=False)
            
            if missing_keys:
                print(f"⚠️ 누락된 키: {missing_keys}")
            if unexpected_keys:
                print(f"⚠️ 예상치 못한 키: {unexpected_keys}")
            
            model.to(self.device)
            model.eval()
            
            print("✅ DiPE 모델 로드 성공!")
            return model
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            
            # 대안: Lightning 모듈로 로드 시도
            try:
                print("🔄 대안 방법 시도...")
                return self.load_lightning_model()
            except Exception as e2:
                print(f"❌ 대안 방법도 실패: {e2}")
                return None

    def load_lightning_model(self):
        """Lightning 모듈로 직접 로드"""
        try:
            from timeprophet.experiments.forecasting import LongTermForecasting
            
            # Lightning 모듈로 로드
            model = LongTermForecasting.load_from_checkpoint(self.model_path)
            model.to(self.device)
            model.eval()
            
            print("✅ Lightning 모듈로 로드 성공!")
            return model.model  # DiPE 모델만 추출
            
        except Exception as e:
            print(f"❌ Lightning 로드 실패: {e}")
            return None

    def load_data(self):
        """데이터 로드"""
        try:
            data = pd.read_csv(self.data_path)
            data['date'] = pd.to_datetime(data['date'])
            data = data.sort_values('date').reset_index(drop=True)
            
            print(f"✅ 데이터 로드 완료: {data.shape}")
            print(f"📅 기간: {data['date'].min()} ~ {data['date'].max()}")
            
            return data
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            return None

    def setup_scaler(self):
        """스케일러 설정"""
        train_data = self.data[
            (self.data['date'] >= '2015-01-01') & 
            (self.data['date'] <= '2022-12-31')
        ]
        
        numeric_cols = [col for col in train_data.columns if col != 'date']
        
        scaler = StandardScaler()
        scaler.fit(train_data[numeric_cols])
        
        print(f"✅ 스케일러 설정 완료: {len(numeric_cols)}개 특성")
        return scaler

    def sliding_window_predict(self, start_date='2023-01-01', end_date='2024-12-31', step_size=7):
        """Sliding Window 예측"""
        
        if self.model is None:
            print("❌ 모델이 로드되지 않아 예측 불가")
            return None
            
        print(f"🔄 Sliding Window 예측 시작")
        print(f"📅 기간: {start_date} ~ {end_date}")
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        all_predictions = []
        current_date = start_dt
        prediction_count = 0
        
        while current_date <= end_dt:
            # 30일 입력 기간
            input_start = current_date - timedelta(days=30)
            input_end = current_date - timedelta(days=1)
            
            # 7일 예측 기간
            pred_start = current_date
            pred_end = current_date + timedelta(days=6)
            
            if prediction_count % 10 == 0:  # 10회마다 출력
                print(f"  📊 예측 {prediction_count+1}: {pred_start.date()} ~ {pred_end.date()}")
            
            # 입력 데이터 추출
            input_data = self.get_input_data(input_start, input_end)
            
            if input_data is not None:
                # 예측 수행
                prediction = self.predict_single_window(input_data)
                
                if prediction is not None:
                    # 예측 날짜들
                    pred_dates = pd.date_range(pred_start, pred_end, freq='D')
                    
                    # 결과 저장
                    for i, date in enumerate(pred_dates):
                        if i < len(prediction):
                            all_predictions.append({
                                'date': date,
                                'prediction_set': prediction_count,
                                'day_ahead': i + 1,
                                'prediction': prediction[i]
                            })
            
            # 다음 윈도우
            current_date += timedelta(days=step_size)
            prediction_count += 1
        
        predictions_df = pd.DataFrame(all_predictions)
        
        print(f"✅ Sliding Window 예측 완료")
        print(f"📊 총 예측 세트: {prediction_count}개")
        print(f"📊 총 예측 포인트: {len(all_predictions)}개")
        
        return predictions_df

    def get_input_data(self, start_date, end_date):
        """30일 입력 데이터 추출"""
        try:
            mask = (self.data['date'] >= start_date) & (self.data['date'] <= end_date)
            period_data = self.data[mask].copy()
            
            if len(period_data) < 30:
                return None
            
            # 정확히 30일
            period_data = period_data.tail(30)
            
            # 수치 데이터만
            numeric_cols = [col for col in period_data.columns if col != 'date']
            input_array = period_data[numeric_cols].values
            
            # 정규화
            input_normalized = self.scaler.transform(input_array)
            
            # 텐서 변환 [1, 30, 402]
            input_tensor = torch.FloatTensor(input_normalized).unsqueeze(0).to(self.device)
            
            return input_tensor
            
        except Exception as e:
            print(f"❌ 입력 데이터 추출 실패: {e}")
            return None

    def predict_single_window(self, input_tensor):
        """단일 윈도우 예측"""
        try:
            with torch.no_grad():
                # DiPE 모델 예측
                output = self.model(input_tensor)
                
                # 넘파이 변환
                prediction = output.cpu().numpy().squeeze(0)
                
                # 역정규화
                prediction_denorm = self.scaler.inverse_transform(prediction)
                
                return prediction_denorm
                
        except Exception as e:
            print(f"❌ 예측 실패: {e}")
            return None

    def save_predictions(self, predictions_df, filename='fixed_dipe_predictions.csv'):
        """예측 결과 저장"""
        if predictions_df is None:
            print("❌ 저장할 예측 결과가 없습니다.")
            return
            
        try:
            # 상세 결과
            detailed_predictions = []
            
            for _, row in predictions_df.iterrows():
                pred_data = row['prediction']
                
                for feature_idx in range(len(pred_data)):
                    detailed_predictions.append({
                        'date': row['date'],
                        'prediction_set': row['prediction_set'],
                        'day_ahead': row['day_ahead'],
                        'feature_idx': feature_idx,
                        'predicted_value': pred_data[feature_idx]
                    })
            
            # 저장
            detailed_df = pd.DataFrame(detailed_predictions)
            detailed_df.to_csv(f'detailed_{filename}', index=False)
            
            # 요약 저장
            summary_df = predictions_df.drop('prediction', axis=1)
            summary_df.to_csv(filename, index=False)
            
            print(f"✅ 예측 결과 저장:")
            print(f"   📄 요약: {filename}")
            print(f"   📄 상세: detailed_{filename}")
            
        except Exception as e:
            print(f"❌ 저장 실패: {e}")

def main():
    """메인 실행"""
    print("🌤️ 수정된 DiPE 모델 2년간 연속 예측")
    print("=" * 50)
    
    # 경로 설정
    model_path = "logs/LongTermForecasting/asos_time_split_30_7/30/7/DiPE/version_1/checkpoints/last.ckpt"
    data_path = "dataset/asos_time_split_30_7.csv.gz"
    
    # 예측기 생성
    predictor = FixedSlidingWindowPredictor(model_path, data_path)
    
    if predictor.model is None:
        print("❌ 모델 로드 실패로 종료")
        return
    
    # Sliding Window 예측 실행
    predictions = predictor.sliding_window_predict(
        start_date='2023-01-01',
        end_date='2024-12-31', 
        step_size=7
    )
    
    # 결과 저장
    predictor.save_predictions(predictions)
    
    print("\n🎉 수정된 2년간 연속 예측 완료!")

if __name__ == "__main__":
    main()