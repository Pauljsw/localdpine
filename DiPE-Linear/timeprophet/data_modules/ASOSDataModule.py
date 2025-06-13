# timeprophet/data_modules/ASOSDataModule.py
"""
ASOS 데이터셋용 커스텀 데이터 모듈
2015-2022: 훈련 데이터
2023-2024: 테스트 데이터
시간순 분할로 현실적 평가
"""

import pandas as pd
from .base import TimeSeriesDataModule

__all__ = ['ASOSDataModule']


class ASOSDataModule(TimeSeriesDataModule):
    """
    ASOS 데이터셋용 커스텀 데이터 모듈
    시간 기반 분할: 2015-2022 (Train) vs 2023-2024 (Test)
    """

    def __read_data__(self) -> pd.DataFrame:
        """ASOS 데이터 로드 (date 컬럼 포함)"""
        df = pd.read_csv(self.dataset_path)
        
        # date 컬럼이 있다면 파싱, 없다면 생성
        if 'date' not in df.columns:
            if all(col in df.columns for col in ['Year', 'Month', 'Day']):
                df['date'] = pd.to_datetime(df[['Year', 'Month', 'Day']], errors='coerce')
            else:
                raise ValueError("날짜 정보가 없습니다. 'date' 컬럼 또는 'Year', 'Month', 'Day' 컬럼이 필요합니다.")
        else:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # 날짜별 정렬
        df = df.sort_values('date').reset_index(drop=True)
        
        # date 컬럼 제거 (시계열 학습용)
        df_final = df.drop('date', axis=1)
        
        # 디버깅 정보
        print(f"📅 ASOS 데이터 기간: {df['date'].min()} ~ {df['date'].max()}")
        print(f"📊 총 데이터 수: {len(df_final)}")
        
        # 연도별 데이터 분포 확인
        year_counts = df['date'].dt.year.value_counts().sort_index()
        print("📈 연도별 데이터 분포:")
        for year, count in year_counts.items():
            print(f"   {year}: {count}일")
        
        return df_final, df['date']  # 데이터와 날짜 정보 반환

    def __split_data__(self, data_and_dates) -> tuple[pd.DataFrame]:
        """
        시간 기반 분할:
        - 훈련: 2015-2022 (8년)
        - 검증: 2022년 후반 일부
        - 테스트: 2023-2024 (2년)
        """
        all_data, dates = data_and_dates
        
        print("🗓️ 시간 기반 데이터 분할 시작...")
        
        # 연도별 인덱스 생성
        train_mask = (dates.dt.year >= 2015) & (dates.dt.year <= 2021)
        val_mask = (dates.dt.year == 2022)
        test_mask = (dates.dt.year >= 2023) & (dates.dt.year <= 2024)
        
        # 기본 분할
        train_indices = dates[train_mask].index
        val_indices = dates[val_mask].index  
        test_indices = dates[test_mask].index
        
        print(f"📊 분할 결과:")
        print(f"   🎯 훈련 데이터: {len(train_indices)}일 (2015-2021)")
        print(f"   🎯 검증 데이터: {len(val_indices)}일 (2022)")
        print(f"   🎯 테스트 데이터: {len(test_indices)}일 (2023-2024)")
        
        # 분할 검증
        if len(train_indices) == 0:
            raise ValueError("훈련 데이터가 없습니다! 2015-2021년 데이터를 확인하세요.")
        if len(test_indices) == 0:
            print("⚠️ 테스트 데이터가 없습니다! 2023-2024년 데이터를 확인하세요.")
        
        # 데이터 추출
        train_data = all_data.iloc[train_indices].copy()
        val_data = all_data.iloc[val_indices].copy() if len(val_indices) > 0 else train_data.tail(365).copy()  # 폴백
        test_data = all_data.iloc[test_indices].copy() if len(test_indices) > 0 else val_data.copy()  # 폴백
        
        # 연속성을 위한 패딩 추가 (시계열 학습 특성상 필요)
        # 검증/테스트 데이터 앞에 input_len만큼의 데이터 추가
        if len(val_indices) > 0:
            val_start_idx = val_indices[0]
            val_padding_start = max(0, val_start_idx - self.input_len)
            val_data = all_data.iloc[val_padding_start:val_indices[-1]+1].copy()
        
        if len(test_indices) > 0:
            test_start_idx = test_indices[0] 
            test_padding_start = max(0, test_start_idx - self.input_len)
            test_data = all_data.iloc[test_padding_start:test_indices[-1]+1].copy()
        
        print(f"✅ 패딩 적용 후:")
        print(f"   📈 훈련: {len(train_data)}일")
        print(f"   📈 검증: {len(val_data)}일")  
        print(f"   📈 테스트: {len(test_data)}일")
        
        return train_data, val_data, test_data

    def prepare_data(self) -> None:
        """데이터 준비 (오버라이드)"""
        print("🔄 ASOS 데이터 준비 중...")
        
        # 데이터 읽기 (커스텀 메서드)
        data_and_dates = self.__read_data__()
        
        # 시간 기반 분할
        train_data, val_data, test_data = self.__split_data__(data_and_dates)
        
        # 다운샘플링
        if self.down_sampling > 1:
            print(f"📉 다운샘플링 적용: {self.down_sampling}")
            train_data = train_data.iloc[::self.down_sampling]
            val_data = val_data.iloc[::self.down_sampling] 
            test_data = test_data.iloc[::self.down_sampling]
        
        # 전처리 적용
        print("🔧 데이터 전처리 중...")
        train_data_processed = self.preprocessor.fit_transform(train_data)
        val_data_processed = self.preprocessor.transform(val_data)
        test_data_processed = self.preprocessor.transform(test_data)
        
        # 텐서 변환
        import torch
        train_data_tensor = torch.from_numpy(train_data_processed).float()
        val_data_tensor = torch.from_numpy(val_data_processed).float()
        test_data_tensor = torch.from_numpy(test_data_processed).float()
        
        # 훈련 비율 적용
        if self.train_proportion < 1.0:
            train_len = int(len(train_data_tensor) * self.train_proportion)
            train_data_tensor = train_data_tensor[:train_len]
            print(f"🎯 훈련 데이터 비율 조정: {self.train_proportion} → {len(train_data_tensor)}일")
        
        # GPU 이동 (필요한 경우)
        if self.gpu:
            train_data_tensor = train_data_tensor.cuda()
            val_data_tensor = val_data_tensor.cuda()
            test_data_tensor = test_data_tensor.cuda()
        
        # 특성 분할
        if self.x_features is None:
            self.train_x = train_data_tensor
            self.val_x = val_data_tensor  
            self.test_x = test_data_tensor
        else:
            self.train_x = train_data_tensor[:, self.x_features]
            self.val_x = val_data_tensor[:, self.x_features]
            self.test_x = test_data_tensor[:, self.x_features]
        
        if self.y_features is None:
            self.train_y = train_data_tensor
            self.val_y = val_data_tensor
            self.test_y = test_data_tensor
        else:
            self.train_y = train_data_tensor[:, self.y_features]
            self.val_y = val_data_tensor[:, self.y_features]
            self.test_y = test_data_tensor[:, self.y_features]
        
        print("✅ ASOS 데이터 준비 완료!")
        print(f"📊 최종 크기 - 훈련: {self.train_x.shape}, 검증: {self.val_x.shape}, 테스트: {self.test_x.shape}")


# timeprophet/data_modules/__init__.py 업데이트
# 기존 파일에 추가
from .ASOSDataModule import ASOSDataModule

__all__ = ['ETDataModule', 'MultivarDataModule', 'ASOSDataModule']