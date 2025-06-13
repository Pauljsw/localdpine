# complete_weather_preprocessing.py
"""
6개 기상요소 모두 포함한 완전 전처리
눈 데이터 특수 처리 포함
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import warnings
warnings.filterwarnings('ignore')

def process_complete_weather_data(input_file='ASOS_Datasett.csv', top_n_stations=95):
    """
    6개 기상요소 모두 포함한 완전 전처리
    """
    print("❄️ 6개 기상요소 완전 전처리 (눈 데이터 포함)")
    print("=" * 60)
    
    # 1. 데이터 로드
    df = pd.read_csv(input_file)
    df['date'] = pd.to_datetime(df[['Year', 'Month', 'Day']], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values(['Loc_ID', 'date']).reset_index(drop=True)
    
    print(f"📊 원본 데이터: {df.shape}")
    print(f"🏠 총 관측소: {df['Loc_ID'].nunique()}개")
    
    # 2. 시간 필터링
    valid_years = (df['date'].dt.year >= 2015) & (df['date'].dt.year <= 2024)
    df = df[valid_years].copy()
    
    # 3. 완전한 6개 기상 요소
    complete_weather_features = [
        'Low_Temp(°C)', 'High_Temp(°C)', 'Precipitation(mm)', 
        'Wind(m/s)', 'Snowfall(cm)', 'Snowdepth(cm)'
    ]
    
    print(f"🌤️ 완전 기상 요소: {complete_weather_features}")
    
    # 4. 관측소 선택
    selected_stations = select_stations_for_complete_weather(df, top_n_stations, complete_weather_features)
    
    # 5. 눈 데이터 특수 전처리
    df = preprocess_snow_data(df, selected_stations)
    
    # 6. 통합 데이터 생성
    unified_data = create_complete_unified_data(df, selected_stations, complete_weather_features)
    
    # 7. 고급 특성 생성
    enhanced_data = create_advanced_weather_features(unified_data, selected_stations, complete_weather_features)
    
    # 8. 최종 정리
    final_data = finalize_complete_data(enhanced_data)
    
    # 9. 저장
    output_file = 'dataset/asos_complete_6weather_30_7.csv.gz'
    final_data.to_csv(output_file, index=False, compression='gzip')
    
    print(f"\n✅ 6개 기상요소 완전 전처리 완료!")
    print(f"📁 저장: {output_file}")
    print(f"📊 최종 크기: {final_data.shape}")
    
    # 10. 눈 데이터 분석
    analyze_snow_data(final_data, selected_stations)
    
    return final_data, selected_stations

def select_stations_for_complete_weather(df: pd.DataFrame, top_n: int, weather_features: List[str]) -> List[int]:
    """6개 기상요소 품질을 고려한 관측소 선택"""
    print(f"🎯 6개 기상요소 품질 기준 관측소 선택...")
    
    station_scores = []
    
    for station_id in df['Loc_ID'].unique():
        station_data = df[df['Loc_ID'] == station_id]
        
        # 훈련/테스트 기간 데이터
        train_data = station_data[(station_data['date'].dt.year >= 2015) & 
                                (station_data['date'].dt.year <= 2022)]
        test_data = station_data[(station_data['date'].dt.year >= 2023) & 
                               (station_data['date'].dt.year <= 2024)]
        
        # 6개 기상요소 완성도 체크
        completeness_scores = []
        for feature in weather_features:
            if feature in station_data.columns:
                # 결측률
                missing_rate = station_data[feature].isnull().sum() / len(station_data)
                
                # 눈 데이터는 특별 처리 (0이 정상값이므로)
                if 'Snow' in feature:
                    # 겨울철 데이터 품질만 체크
                    winter_data = station_data[
                        (station_data['date'].dt.month.isin([11, 12, 1, 2, 3]))
                    ]
                    if len(winter_data) > 0:
                        winter_missing = winter_data[feature].isnull().sum() / len(winter_data)
                        completeness_scores.append(1 - winter_missing)
                    else:
                        completeness_scores.append(0.5)  # 겨울 데이터 없음
                else:
                    completeness_scores.append(1 - missing_rate)
            else:
                completeness_scores.append(0)
        
        # 종합 점수
        avg_completeness = np.mean(completeness_scores)
        
        # 데이터 양 점수
        train_score = len(train_data) / (8 * 365)
        test_score = len(test_data) / (2 * 365) if len(test_data) > 0 else 0
        
        total_score = (
            avg_completeness * 0.4 +    # 6개 요소 완성도 40%
            train_score * 0.4 +         # 훈련 데이터 40%
            test_score * 0.2            # 테스트 데이터 20%
        )
        
        station_scores.append({
            'station_id': station_id,
            'location': station_data['Location'].iloc[0] if len(station_data) > 0 else 'Unknown',
            'latitude': station_data['Latitude'].iloc[0] if len(station_data) > 0 else np.nan,
            'elevation': station_data['Elevation'].iloc[0] if len(station_data) > 0 else np.nan,
            'train_days': len(train_data),
            'test_days': len(test_data),
            'weather_completeness': avg_completeness,
            'total_score': total_score
        })
    
    # 점수순 정렬 및 선택
    scores_df = pd.DataFrame(station_scores).sort_values('total_score', ascending=False)
    
    # 최소 요구사항 (3년 훈련 데이터)
    qualified = scores_df[scores_df['train_days'] >= 365 * 3]
    
    if len(qualified) < top_n:
        top_n = len(qualified)
        print(f"⚠️ 요구사항 만족 관측소: {top_n}개")
    
    selected = qualified.head(top_n)['station_id'].tolist()
    
    print(f"📍 선택된 {len(selected)}개 관측소 (6개 기상요소 기준):")
    for i, station_id in enumerate(selected[:10]):  # 상위 10개만 표시
        info = scores_df[scores_df['station_id'] == station_id].iloc[0]
        print(f"  {i+1:2d}. 관측소 {station_id} ({info['location'][:15]})")
        print(f"      완성도: {info['weather_completeness']:.3f}, 점수: {info['total_score']:.3f}")
    
    if len(selected) > 10:
        print(f"      ... 외 {len(selected)-10}개 관측소")
    
    return selected

def preprocess_snow_data(df: pd.DataFrame, stations: List[int]) -> pd.DataFrame:
    """눈 데이터 특수 전처리"""
    print("❄️ 눈 데이터 특수 전처리...")
    
    enhanced_df = df.copy()
    
    # 계절 정보 추가
    enhanced_df['Month'] = enhanced_df['date'].dt.month
    enhanced_df['Is_Winter'] = ((enhanced_df['Month'] >= 11) | (enhanced_df['Month'] <= 3)).astype(int)
    enhanced_df['Winter_Intensity'] = enhanced_df['Is_Winter'] * (4 - np.abs(enhanced_df['Month'] - 1))
    
    # 관측소별 눈 데이터 처리
    for station_id in stations:
        station_mask = enhanced_df['Loc_ID'] == station_id
        station_data = enhanced_df[station_mask].copy()
        
        if len(station_data) == 0:
            continue
        
        # 1. 눈 데이터 결측값 처리 (계절 고려)
        for snow_col in ['Snowfall(cm)', 'Snowdepth(cm)']:
            if snow_col in station_data.columns:
                # 여름철(6-9월) 결측값은 0으로
                summer_mask = station_data['Month'].isin([6, 7, 8, 9])
                enhanced_df.loc[station_mask & summer_mask, snow_col] = enhanced_df.loc[station_mask & summer_mask, snow_col].fillna(0)
                
                # 겨울철 결측값은 선형 보간
                winter_mask = station_data['Is_Winter'] == 1
                if winter_mask.any():
                    enhanced_df.loc[station_mask & winter_mask, snow_col] = enhanced_df.loc[station_mask & winter_mask, snow_col].interpolate(method='linear')
                
                # 나머지 결측값은 전후 값으로
                enhanced_df.loc[station_mask, snow_col] = enhanced_df.loc[station_mask, snow_col].fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        # 2. 눈 데이터 이상값 처리 (물리적 한계)
        if 'Snowfall(cm)' in enhanced_df.columns:
            enhanced_df.loc[station_mask, 'Snowfall(cm)'] = enhanced_df.loc[station_mask, 'Snowfall(cm)'].clip(0, 100)  # 일 적설량 최대 100cm
        
        if 'Snowdepth(cm)' in enhanced_df.columns:
            enhanced_df.loc[station_mask, 'Snowdepth(cm)'] = enhanced_df.loc[station_mask, 'Snowdepth(cm)'].clip(0, 300)  # 적설 깊이 최대 300cm
    
    print("✅ 눈 데이터 특수 전처리 완료")
    return enhanced_df

def create_complete_unified_data(df: pd.DataFrame, stations: List[int], features: List[str]) -> pd.DataFrame:
    """6개 기상요소 통합 데이터 생성"""
    print(f"🌍 6개 기상요소 통합 데이터 생성...")
    
    # 전체 날짜 범위
    min_date = df['date'].min()
    max_date = df['date'].max()
    full_date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    
    # 통합 데이터프레임 초기화
    unified_data = pd.DataFrame({'date': full_date_range})
    
    # 각 관측소별 6개 기상요소 추가
    for i, station_id in enumerate(stations):
        if i % 20 == 0:  # 20개마다 진행상황 출력
            print(f"  📡 관측소 {station_id} 처리... ({i+1}/{len(stations)})")
        
        station_data = df[df['Loc_ID'] == station_id][['date'] + features].copy()
        
        # 컬럼명 변경 (6개 요소 모두)
        rename_dict = {}
        for feature in features:
            clean_name = feature.split('(')[0].replace(' ', '').replace('_', '')
            rename_dict[feature] = f"{clean_name}_S{station_id}"
        
        station_data = station_data.rename(columns=rename_dict)
        
        # 날짜별 병합
        unified_data = pd.merge(unified_data, station_data, on='date', how='left')
    
    # 결측값 처리
    print("🔧 통합 데이터 결측값 처리...")
    numeric_cols = [col for col in unified_data.columns if col != 'date']
    
    # 선형 보간
    unified_data = unified_data.sort_values('date')
    unified_data[numeric_cols] = unified_data[numeric_cols].interpolate(method='linear')
    unified_data[numeric_cols] = unified_data[numeric_cols].fillna(method='bfill').fillna(method='ffill')
    
    print(f"✅ 6개 기상요소 통합 완료: {unified_data.shape}")
    return unified_data

def create_advanced_weather_features(data: pd.DataFrame, stations: List[int], base_features: List[str]) -> pd.DataFrame:
    """고급 기상 특성 생성 (눈 포함)"""
    print("⚡ 고급 기상 특성 생성...")
    
    enhanced = data.copy()
    
    # 1. 기본 시간 특성
    enhanced['Year'] = enhanced['date'].dt.year
    enhanced['Month'] = enhanced['date'].dt.month
    enhanced['DayOfYear'] = enhanced['date'].dt.dayofyear
    enhanced['Season'] = ((enhanced['Month'] - 1) // 3) + 1
    enhanced['Is_Winter'] = ((enhanced['Month'] >= 11) | (enhanced['Month'] <= 3)).astype(int)
    
    # 2. 순환적 인코딩
    enhanced['Month_Sin'] = np.sin(2 * np.pi * enhanced['Month'] / 12)
    enhanced['Month_Cos'] = np.cos(2 * np.pi * enhanced['Month'] / 12)
    enhanced['Day_Sin'] = np.sin(2 * np.pi * enhanced['DayOfYear'] / 365)
    enhanced['Day_Cos'] = np.cos(2 * np.pi * enhanced['DayOfYear'] / 365)
    
    # 3. 전국 통계 (6개 요소 모두)
    for weather_type in ['LowTemp', 'HighTemp', 'Precipitation', 'Wind', 'Snowfall', 'Snowdepth']:
        var_cols = [col for col in data.columns if weather_type in col and 'S' in col]
        
        if var_cols:
            enhanced[f'National_Avg_{weather_type}'] = enhanced[var_cols].mean(axis=1)
            enhanced[f'National_Max_{weather_type}'] = enhanced[var_cols].max(axis=1)
            enhanced[f'National_Min_{weather_type}'] = enhanced[var_cols].min(axis=1)
            
            if weather_type in ['LowTemp', 'HighTemp']:
                enhanced[f'National_{weather_type}_Range'] = enhanced[f'National_Max_{weather_type}'] - enhanced[f'National_Min_{weather_type}']
    
    # 4. 눈 특수 특성
    snow_cols = [col for col in data.columns if 'Snowfall' in col]
    depth_cols = [col for col in data.columns if 'Snowdepth' in col]
    
    if snow_cols:
        # 전국 눈 오는 관측소 수
        enhanced['Snow_Station_Count'] = (enhanced[snow_cols] > 0).sum(axis=1)
        enhanced['Snow_Coverage_Ratio'] = enhanced['Snow_Station_Count'] / len(snow_cols)
        
        # 눈 강도 분류
        enhanced['Heavy_Snow_Stations'] = (enhanced[snow_cols] > 5).sum(axis=1)  # 5cm 이상
        enhanced['Extreme_Snow_Stations'] = (enhanced[snow_cols] > 20).sum(axis=1)  # 20cm 이상
    
    if depth_cols:
        # 적설 지속성
        enhanced['Snow_Depth_Persistence'] = (enhanced[depth_cols] > 0).sum(axis=1)
        enhanced['Deep_Snow_Stations'] = (enhanced[depth_cols] > 10).sum(axis=1)  # 10cm 이상 적설
    
    # 5. 온도 관련 고급 특성
    if 'National_Avg_LowTemp' in enhanced.columns and 'National_Avg_HighTemp' in enhanced.columns:
        enhanced['National_Daily_Temp_Range'] = enhanced['National_Avg_HighTemp'] - enhanced['National_Avg_LowTemp']
        enhanced['National_Avg_Temp'] = (enhanced['National_Avg_HighTemp'] + enhanced['National_Avg_LowTemp']) / 2
        
        # 한파/폭염 지수
        enhanced['Cold_Wave_Index'] = np.maximum(0, -5 - enhanced['National_Avg_LowTemp'])  # -5도 이하
        enhanced['Heat_Wave_Index'] = np.maximum(0, enhanced['National_Avg_HighTemp'] - 30)  # 30도 이상
    
    # 6. 이동 평균 (전체 기상요소)
    for window in [3, 7, 14]:
        for avg_col in ['National_Avg_LowTemp', 'National_Avg_HighTemp', 'National_Avg_Precipitation', 
                       'National_Avg_Wind', 'National_Avg_Snowfall', 'National_Avg_Snowdepth']:
            if avg_col in enhanced.columns:
                enhanced[f'{avg_col}_MA{window}'] = enhanced[avg_col].rolling(window, min_periods=1).mean()
    
    print(f"✅ 고급 특성 생성 완료: {len(enhanced.columns)}개 특성")
    return enhanced

def finalize_complete_data(data: pd.DataFrame) -> pd.DataFrame:
    """최종 데이터 정리"""
    print("🔍 최종 데이터 정리...")
    
    # 무한값 처리
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].replace([np.inf, -np.inf], np.nan)
    data[numeric_cols] = data[numeric_cols].fillna(method='bfill').fillna(method='ffill')
    
    # date 컬럼을 맨 앞으로
    cols = ['date'] + [col for col in data.columns if col != 'date']
    data = data[cols]
    
    print(f"✅ 최종 정리 완료: {data.shape}")
    return data

def analyze_snow_data(data: pd.DataFrame, stations: List[int]):
    """눈 데이터 분석"""
    print("\n❄️ 눈 데이터 분석")
    print("=" * 30)
    
    # 눈 관련 컬럼 찾기
    snow_cols = [col for col in data.columns if 'Snowfall' in col]
    depth_cols = [col for col in data.columns if 'Snowdepth' in col]
    
    if snow_cols:
        # 전국 눈 통계
        snow_data = data[snow_cols].values.flatten()
        snow_nonzero = snow_data[snow_data > 0]
        
        print(f"❄️ 적설량 통계:")
        print(f"   📊 전체 데이터 포인트: {len(snow_data):,}개")
        print(f"   📊 눈 오는 경우: {len(snow_nonzero):,}개 ({len(snow_nonzero)/len(snow_data)*100:.1f}%)")
        if len(snow_nonzero) > 0:
            print(f"   📊 평균 적설량: {np.mean(snow_nonzero):.2f}cm")
            print(f"   📊 최대 적설량: {np.max(snow_nonzero):.2f}cm")
    
    if depth_cols:
        depth_data = data[depth_cols].values.flatten()
        depth_nonzero = depth_data[depth_data > 0]
        
        print(f"🏔️ 적설 깊이 통계:")
        print(f"   📊 적설 있는 경우: {len(depth_nonzero):,}개 ({len(depth_nonzero)/len(depth_data)*100:.1f}%)")
        if len(depth_nonzero) > 0:
            print(f"   📊 평균 적설 깊이: {np.mean(depth_nonzero):.2f}cm")
            print(f"   📊 최대 적설 깊이: {np.max(depth_nonzero):.2f}cm")
    
    # 계절별 눈 분석
    winter_data = data[data['Is_Winter'] == 1]
    if len(winter_data) > 0 and snow_cols:
        winter_snow = winter_data[snow_cols].values.flatten()
        winter_snow_nonzero = winter_snow[winter_snow > 0]
        
        print(f"🗓️ 겨울철 눈 통계:")
        print(f"   📊 겨울철 눈 비율: {len(winter_snow_nonzero)/len(winter_snow)*100:.1f}%")
        if len(winter_snow_nonzero) > 0:
            print(f"   📊 겨울철 평균 적설량: {np.mean(winter_snow_nonzero):.2f}cm")

def main():
    """메인 실행"""
    print("❄️ 6개 기상요소 완전 전처리 시작")
    print("🌤️ Low_Temp, High_Temp, Precipitation, Wind, Snowfall, Snowdepth")
    print("=" * 80)
    
    # 완전 전처리 실행
    final_data, stations = process_complete_weather_data('ASOS_Datasett.csv', top_n_stations=95)
    
    print(f"\n🎉 6개 기상요소 완전 전처리 완료!")
    print(f"🎯 선택된 관측소: {len(stations)}개")
    print(f"📊 특성 수: {len(final_data.columns)}개 (date 포함)")
    print(f"❄️ 눈 데이터 포함으로 겨울철 예측 정확도 향상 기대!")
    print(f"🚀 완전한 기상 예측 모델 준비 완료!")

if __name__ == "__main__":
    main()