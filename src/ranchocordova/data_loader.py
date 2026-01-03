"""
Centralized Data Loader for Enhanced Energy Data
================================================

Loads and manages all energy-related datasets:
- Hourly consumption (8760 hours × 100 accounts)
- TOU rates
- Rebates
- Benchmarks
- Equipment UEC
- Annual reports
"""

import os
import warnings
from typing import Dict, Optional

import pandas as pd

warnings.filterwarnings("ignore")


class EnergyDataLoader:
    """Singleton data loader for all energy datasets"""

    _instance = None
    _data_loaded = False

    # Data storage
    original_energy_df = None
    customer_service_df = None
    hourly_consumption_df = None
    tou_rates_df = None
    rebates_df = None
    benchmarks_df = None
    equipment_uec_utility_df = None
    equipment_uec_dwelling_df = None
    annual_report_df = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize data loader (loads data only once)"""
        if not self._data_loaded:
            self.load_all_data()

    def load_all_data(self):
        """Load all datasets from data directory"""
        print("📊 Loading energy datasets...")

        base_path = os.path.join(os.path.dirname(__file__), "data")

        try:
            # Original data (required)
            self.original_energy_df = self._load_csv(
                base_path, "Energy.txt", required=True
            )
            self.customer_service_df = self._load_csv(
                base_path, "CustomerService.txt", required=True
            )

            # New datasets (optional - gracefully handle missing files)
            self.hourly_consumption_df = self._load_csv(
                base_path,
                "AI-Data-2025-R1.csv",
                description="Hourly consumption data (8760 hours)",
            )

            self.tou_rates_df = self._load_csv(
                base_path, "SMUD_TOU_Rates.csv", description="Time-of-Use rate tables"
            )

            self.rebates_df = self._load_csv(
                base_path, "SMUD_Rebates.csv", description="Rebate programs"
            )

            self.benchmarks_df = self._load_csv(
                base_path,
                "CA_Benchmarks.csv",
                description="California utility benchmarks",
            )

            self.equipment_uec_utility_df = self._load_csv(
                base_path,
                "Equipment_UEC_Utility.csv",
                description="Equipment UEC by utility (Table 1)",
            )

            self.equipment_uec_dwelling_df = self._load_csv(
                base_path,
                "Equipment_UEC_Dwelling.csv",
                description="Equipment UEC by dwelling type (Table 2)",
            )

            self.annual_report_df = self._load_csv(
                base_path,
                "SMUD_Annual_Report.csv",
                description="SMUD annual report data",
            )

            self._data_loaded = True
            print("✅ Energy datasets loaded successfully")
            self._print_summary()

        except Exception as e:
            print(f"❌ Error loading datasets: {e}")
            raise

    def _load_csv(
        self,
        base_path: str,
        filename: str,
        required: bool = False,
        description: str = None,
    ) -> Optional[pd.DataFrame]:
        """Load a single CSV file with error handling"""
        filepath = os.path.join(base_path, filename)

        if not os.path.exists(filepath):
            if required:
                raise FileNotFoundError(f"Required file not found: {filename}")
            else:
                print(f"  ⚠️  {filename} not found (optional)")
                return None

        try:
            df = pd.read_csv(filepath)
            desc = description or filename
            print(f"  ✓ Loaded {filename}: {len(df)} rows")
            return df
        except Exception as e:
            if required:
                raise Exception(f"Error loading {filename}: {e}")
            else:
                print(f"  ⚠️  Could not load {filename}: {e}")
                return None

    def _print_summary(self):
        """Print summary of loaded data"""
        print("\n📈 Dataset Summary:")
        print(
            f"  Original Energy: {len(self.original_energy_df) if self.original_energy_df is not None else 0} records"
        )
        print(
            f"  Customer Service: {len(self.customer_service_df) if self.customer_service_df is not None else 0} calls"
        )

        if self.hourly_consumption_df is not None:
            print(f"  Hourly Data: {len(self.hourly_consumption_df)} hours")
        if self.tou_rates_df is not None:
            print(f"  TOU Rates: {len(self.tou_rates_df)} rate periods")
        if self.rebates_df is not None:
            print(f"  Rebates: {len(self.rebates_df)} programs")
        if self.benchmarks_df is not None:
            print(f"  Benchmarks: {len(self.benchmarks_df)} utilities")
        if self.equipment_uec_utility_df is not None:
            print(
                f"  Equipment UEC (Utility): {len(self.equipment_uec_utility_df)} items"
            )
        if self.equipment_uec_dwelling_df is not None:
            print(
                f"  Equipment UEC (Dwelling): {len(self.equipment_uec_dwelling_df)} items"
            )
        if self.annual_report_df is not None:
            print(f"  Annual Report: {len(self.annual_report_df)} entries")
        print()

    def get_available_datasets(self) -> Dict[str, bool]:
        """Return which datasets are available"""
        return {
            "original_energy": self.original_energy_df is not None,
            "customer_service": self.customer_service_df is not None,
            "hourly_consumption": self.hourly_consumption_df is not None,
            "tou_rates": self.tou_rates_df is not None,
            "rebates": self.rebates_df is not None,
            "benchmarks": self.benchmarks_df is not None,
            "equipment_uec_utility": self.equipment_uec_utility_df is not None,
            "equipment_uec_dwelling": self.equipment_uec_dwelling_df is not None,
            "annual_report": self.annual_report_df is not None,
        }

    def reload(self):
        """Force reload all data"""
        self._data_loaded = False
        self.load_all_data()


# Global instance
_data_loader = None


def get_data_loader() -> EnergyDataLoader:
    """Get singleton instance of data loader"""
    global _data_loader
    if _data_loader is None:
        _data_loader = EnergyDataLoader()
    return _data_loader


# Convenience functions
def get_hourly_data() -> Optional[pd.DataFrame]:
    """Get hourly consumption data"""
    return get_data_loader().hourly_consumption_df


def get_tou_rates() -> Optional[pd.DataFrame]:
    """Get TOU rate data"""
    return get_data_loader().tou_rates_df


def get_rebates() -> Optional[pd.DataFrame]:
    """Get rebate programs data"""
    return get_data_loader().rebates_df


def get_benchmarks() -> Optional[pd.DataFrame]:
    """Get benchmark data"""
    return get_data_loader().benchmarks_df


def get_equipment_uec_utility() -> Optional[pd.DataFrame]:
    """Get equipment UEC by utility"""
    return get_data_loader().equipment_uec_utility_df


def get_equipment_uec_dwelling() -> Optional[pd.DataFrame]:
    """Get equipment UEC by dwelling type"""
    return get_data_loader().equipment_uec_dwelling_df


def get_annual_report() -> Optional[pd.DataFrame]:
    """Get annual report data"""
    return get_data_loader().annual_report_df


# if __name__ == "__main__":
#     """Test the data loader"""
#     print("Testing Energy Data Loader")
#     print("=" * 60)

#     loader = get_data_loader()

#     print("\nAvailable datasets:")
#     for dataset, available in loader.get_available_datasets().items():
#         status = "✓" if available else "✗"
#         print(f"  {status} {dataset}")

#     print("\n" + "=" * 60)
#     print("Data loader test complete!")
