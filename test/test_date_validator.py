import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.date_validator import DateValidator

EXPECTED_AD = {
    "count": 1,
    "date": {
        "production": None,
        "expiration": {"year": 2026, "month": 5, "day": 2},
    },
}
EXPECTED_MINGUO = {
    "count": 1,
    "date": {
        "production": None,
        "expiration": {"year": 2026, "month": 5, "day": 2},
    },
}  # 115 + 1911 = 2026


class TestExtractDateAD_YMD:
    """西元年月日格式測試"""

    def test_dash_separator(self):
        assert DateValidator.extract_expiry_date("2026-05-02") == EXPECTED_AD

    def test_dot_separator(self):
        assert DateValidator.extract_expiry_date("2026.05.02") == EXPECTED_AD

    def test_slash_separator(self):
        assert DateValidator.extract_expiry_date("2026/05/02") == EXPECTED_AD

    def test_space_separator(self):
        assert DateValidator.extract_expiry_date("2026 05 02") == EXPECTED_AD

    def test_no_separator(self):
        assert DateValidator.extract_expiry_date("20260502") == EXPECTED_AD


class TestExtractDateAD_DMY:
    """西元日月年格式測試"""

    def test_dash_separator(self):
        assert DateValidator.extract_expiry_date("02-05-2026") == EXPECTED_AD

    def test_dot_separator(self):
        assert DateValidator.extract_expiry_date("02.05.2026") == EXPECTED_AD

    def test_slash_separator(self):
        assert DateValidator.extract_expiry_date("02/05/2026") == EXPECTED_AD

    def test_space_separator(self):
        assert DateValidator.extract_expiry_date("02 05 2026") == EXPECTED_AD

    def test_no_separator(self):
        assert DateValidator.extract_expiry_date("02052026") == EXPECTED_AD


# class TestExtractDateAD_Partial:
#     """西元年格式部分缺損測試"""

#     def test_dash_separator(self):
#         assert DateValidator.extract_expiry_date("2026-0502") == EXPECTED_AD


class TestExtractDateAD_Invalid:
    """西元年格式無效日期測試"""

    def test_invalid_month(self):
        result = DateValidator.extract_expiry_date("2026-13-01")
        assert result["count"] == 0

    def test_invalid_day(self):
        result = DateValidator.extract_expiry_date("2026-02-30")
        assert result["count"] == 0

    def test_invalid_day(self):
        result = DateValidator.extract_expiry_date("2026-02-29")
        assert result["count"] == 0


class TestExtractDateMinguo:
    """民國年格式測試"""

    def test_dash_separator(self):
        assert DateValidator.extract_expiry_date("115-05-02") == EXPECTED_MINGUO

    def test_dot_separator(self):
        assert DateValidator.extract_expiry_date("115.05.02") == EXPECTED_MINGUO

    def test_slash_separator(self):
        assert DateValidator.extract_expiry_date("115/05/02") == EXPECTED_MINGUO

    def test_space_separator(self):
        assert DateValidator.extract_expiry_date("115 05 02") == EXPECTED_MINGUO

    def test_no_separator(self):
        assert DateValidator.extract_expiry_date("1150502") == EXPECTED_MINGUO


class TestExtractDateEngMonth_DMMY:
    """英文月份格式 DD MMM YY 測試"""

    def test_space_separator(self):
        assert DateValidator.extract_expiry_date("02 MAY 26") == EXPECTED_AD

    def test_dash_separator(self):
        assert DateValidator.extract_expiry_date("02-MAY-26") == EXPECTED_AD

    def test_dot_separator(self):
        assert DateValidator.extract_expiry_date("02.MAY.26") == EXPECTED_AD

    def test_slash_separator(self):
        assert DateValidator.extract_expiry_date("02/MAY/26") == EXPECTED_AD

    def test_lowercase(self):
        assert DateValidator.extract_expiry_date("02 may 26") == EXPECTED_AD


class TestExtractDateEngMonth_DMMYYYY:
    """英文月份格式 DD MMM YYYY 測試"""

    def test_space_separator(self):
        assert DateValidator.extract_expiry_date("02 MAY 2026") == EXPECTED_AD

    def test_dash_separator(self):
        assert DateValidator.extract_expiry_date("02-MAY-2026") == EXPECTED_AD

    def test_dot_separator(self):
        assert DateValidator.extract_expiry_date("02.MAY.2026") == EXPECTED_AD

    def test_slash_separator(self):
        assert DateValidator.extract_expiry_date("02/MAY/2026") == EXPECTED_AD


class TestExtractDateEngMonth_YYYYMMD:
    """英文月份格式 YYYY MMM DD 測試"""

    def test_space_separator(self):
        assert DateValidator.extract_expiry_date("2026 MAY 02") == EXPECTED_AD

    def test_dash_separator(self):
        assert DateValidator.extract_expiry_date("2026-MAY-02") == EXPECTED_AD

    def test_dot_separator(self):
        assert DateValidator.extract_expiry_date("2026.MAY.02") == EXPECTED_AD

    def test_slash_separator(self):
        assert DateValidator.extract_expiry_date("2026/MAY/02") == EXPECTED_AD


class TestExtractDateEngMonth_MMMYYYY:
    """英文月份格式 MMM YYYY 測試 (day=1)"""

    EXPECTED = {
        "count": 1,
        "date": {
            "production": None,
            "expiration": {"year": 2026, "month": 5, "day": 1},
        },
    }

    def test_space_separator(self):
        assert DateValidator.extract_expiry_date("MAY 2026") == self.EXPECTED

    def test_dash_separator(self):
        assert DateValidator.extract_expiry_date("MAY-2026") == self.EXPECTED

    def test_dot_separator(self):
        assert DateValidator.extract_expiry_date("MAY.2026") == self.EXPECTED

    def test_slash_separator(self):
        assert DateValidator.extract_expiry_date("MAY/2026") == self.EXPECTED

    def test_two_digit_year(self):
        assert DateValidator.extract_expiry_date("MAY 26") == self.EXPECTED


class TestExtractDateEngMonth_MMMDDYY:
    """英文月份格式 MMM DD YY 測試"""

    def test_space_separator(self):
        assert DateValidator.extract_expiry_date("MAY 02 26") == EXPECTED_AD

    def test_dash_separator(self):
        assert DateValidator.extract_expiry_date("MAY-02-26") == EXPECTED_AD

    def test_dot_separator(self):
        assert DateValidator.extract_expiry_date("MAY.02.26") == EXPECTED_AD

    def test_slash_separator(self):
        assert DateValidator.extract_expiry_date("MAY/02/26") == EXPECTED_AD


class TestExtractDateEngMonth_MMMDDYYYY:
    """英文月份格式 MMM DD YYYY 測試"""

    def test_space_separator(self):
        assert DateValidator.extract_expiry_date("MAY 02 2026") == EXPECTED_AD

    def test_dash_separator(self):
        assert DateValidator.extract_expiry_date("MAY-02-2026") == EXPECTED_AD

    def test_dot_separator(self):
        assert DateValidator.extract_expiry_date("MAY.02.2026") == EXPECTED_AD

    def test_slash_separator(self):
        assert DateValidator.extract_expiry_date("MAY/02/2026") == EXPECTED_AD


class TestExtractDateYYMMDD:
    """兩位數年份格式 YY MM DD 測試"""

    def test_space_separator(self):
        assert DateValidator.extract_expiry_date("26 05 02") == EXPECTED_AD

    def test_dash_separator(self):
        assert DateValidator.extract_expiry_date("26-05-02") == EXPECTED_AD

    def test_dot_separator(self):
        assert DateValidator.extract_expiry_date("26.05.02") == EXPECTED_AD

    def test_slash_separator(self):
        assert DateValidator.extract_expiry_date("26/05/02") == EXPECTED_AD


class TestExtractDateYYYYMM:
    """年月格式 YYYY MM 測試 (day=1)"""

    EXPECTED = {
        "count": 1,
        "date": {
            "production": None,
            "expiration": {"year": 2026, "month": 5, "day": 1},
        },
    }

    def test_space_separator(self):
        assert DateValidator.extract_expiry_date("2026 05") == self.EXPECTED

    def test_dash_separator(self):
        assert DateValidator.extract_expiry_date("2026-05") == self.EXPECTED

    def test_dot_separator(self):
        assert DateValidator.extract_expiry_date("2026.05") == self.EXPECTED

    def test_slash_separator(self):
        assert DateValidator.extract_expiry_date("2026/05") == self.EXPECTED


class TestExtractDateMMYYYY:
    """月年格式 MM YYYY 測試 (day=1)"""

    EXPECTED = {
        "count": 1,
        "date": {
            "production": None,
            "expiration": {"year": 2026, "month": 5, "day": 1},
        },
    }

    def test_space_separator(self):
        assert DateValidator.extract_expiry_date("05 2026") == self.EXPECTED

    def test_dash_separator(self):
        assert DateValidator.extract_expiry_date("05-2026") == self.EXPECTED

    def test_dot_separator(self):
        assert DateValidator.extract_expiry_date("05.2026") == self.EXPECTED

    def test_slash_separator(self):
        assert DateValidator.extract_expiry_date("05/2026") == self.EXPECTED


class TestExtractDateMMDD:
    """月日格式 MM DD 測試 (year=當前年份)"""

    current_year = datetime.now().year

    EXPECTED = {
        "count": 1,
        "date": {
            "production": None,
            "expiration": {"year": current_year, "month": 5, "day": 2},
        },
    }

    def test_space_separator(self):
        assert DateValidator.extract_expiry_date("05 02") == self.EXPECTED

    def test_dash_separator(self):
        assert DateValidator.extract_expiry_date("05-02") == self.EXPECTED

    def test_dot_separator(self):
        assert DateValidator.extract_expiry_date("05.02") == self.EXPECTED

    def test_slash_separator(self):
        assert DateValidator.extract_expiry_date("05/02") == self.EXPECTED


class TestExtractDateInvalid:
    """無效日期測試"""

    def test_invalid_month(self):
        result = DateValidator.extract_expiry_date("2026-13-01")
        assert result["count"] == 0

    def test_invalid_day(self):
        result = DateValidator.extract_expiry_date("2026-02-30")
        assert result["count"] == 0

    def test_no_date(self):
        result = DateValidator.extract_expiry_date("hello world")
        assert result["count"] == 0


class TestExtractMultipleDates:
    """多日期格式測試 (.PD 製造日期, .BB 有效日期)"""

    def test_pd_and_bb(self):
        """測試同時有 .PD 和 .BB"""
        texts = [".F25226B 04:49", ".PD: 14 / 08/2025", ".BB: 14 / 08/2026"]
        text = " ".join(texts)
        result = DateValidator.extract_multiple_dates(text)
        expected = {
            "count": 2,
            "date": {
                "production": {"year": 2025, "month": 8, "day": 14},
                "expiration": {"year": 2026, "month": 8, "day": 14},
            },
        }
        assert result == expected

    def test_only_pd(self):
        """測試只有 .PD"""
        text = ".F25226B 04:49 .PD: 14 / 08/2025"
        result = DateValidator.extract_multiple_dates(text)
        expected = {
            "count": 1,
            "date": {
                "production": {"year": 2025, "month": 8, "day": 14},
                "expiration": None,
            },
        }
        assert result == expected

    def test_only_bb(self):
        """測試只有 .BB"""
        text = ".F25226B 04:49 .BB: 14 / 08/2026"
        result = DateValidator.extract_multiple_dates(text)
        expected = {
            "count": 1,
            "date": {
                "production": None,
                "expiration": {"year": 2026, "month": 8, "day": 14},
            },
        }
        assert result == expected

    def test_no_pd_no_bb(self):
        """測試沒有 .PD 和 .BB"""
        text = ".F25226B 04:49 some random text"
        result = DateValidator.extract_multiple_dates(text)
        expected = {"count": 0, "date": None}
        assert result == expected

    def test_chinese_pd_and_bb(self):
        """測試中文關鍵字 製造 和 有效"""
        text = "製造日期: 2025/08/14 有效日期: 2026/08/14"
        result = DateValidator.extract_multiple_dates(text)
        expected = {
            "count": 2,
            "date": {
                "production": {"year": 2025, "month": 8, "day": 14},
                "expiration": {"year": 2026, "month": 8, "day": 14},
            },
        }
        assert result == expected

    def test_chinese_only_pd(self):
        """測試只有中文 製造"""
        text = "製造日期: 2025/08/14"
        result = DateValidator.extract_multiple_dates(text)
        expected = {
            "count": 1,
            "date": {
                "production": {"year": 2025, "month": 8, "day": 14},
                "expiration": None,
            },
        }
        assert result == expected

    def test_chinese_only_bb(self):
        """測試只有中文 有效"""
        text = "有效日期: 2026/08/14"
        result = DateValidator.extract_multiple_dates(text)
        expected = {
            "count": 1,
            "date": {
                "production": None,
                "expiration": {"year": 2026, "month": 8, "day": 14},
            },
        }
        assert result == expected

    def test_no_keyword_two_dates_older_first(self):
        """測試無關鍵字，兩個日期 (較舊在前)"""
        text = "2025/08/14 2026/08/14"
        result = DateValidator.extract_multiple_dates(text)
        expected = {
            "count": 2,
            "date": {
                "production": {"year": 2025, "month": 8, "day": 14},
                "expiration": {"year": 2026, "month": 8, "day": 14},
            },
        }
        assert result == expected

    def test_no_keyword_two_dates_newer_first(self):
        """測試無關鍵字，兩個日期 (較新在前)，自動比較交換"""
        text = "2026/08/14 2025/08/14"
        result = DateValidator.extract_multiple_dates(text)
        expected = {
            "count": 2,
            "date": {
                "production": {"year": 2025, "month": 8, "day": 14},
                "expiration": {"year": 2026, "month": 8, "day": 14},
            },
        }
        assert result == expected

    def test_no_keyword_single_date(self):
        """測試無關鍵字，只有一個日期，預設為有效日期"""
        text = "2026/08/14"
        result = DateValidator.extract_multiple_dates(text)
        expected = {
            "count": 1,
            "date": {
                "production": None,
                "expiration": {"year": 2026, "month": 8, "day": 14},
            },
        }
        assert result == expected

    def test_no_keyword_two_dates_with_spaces(self):
        """測試無關鍵字，日期有空格分隔符"""
        text = "14 / 08 / 2025 14 / 08 / 2026"
        result = DateValidator.extract_multiple_dates(text)
        expected = {
            "count": 2,
            "date": {
                "production": {"year": 2025, "month": 8, "day": 14},
                "expiration": {"year": 2026, "month": 8, "day": 14},
            },
        }
        assert result == expected
