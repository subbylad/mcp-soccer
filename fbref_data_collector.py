#!/usr/bin/env python3
"""
Comprehensive FBref Data Collector for MCP Soccer Server
Downloads all available player statistics from Big 5 European leagues
"""

import soccerdata as sd
import pandas as pd
import numpy as np
from pathlib import Path
import time
from datetime import datetime
import logging
from typing import Dict, List, Optional
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveFBrefCollector:
    """Comprehensive FBref data collector for MCP Soccer Server"""
    
    def __init__(self, seasons: List[str] = None):
        if seasons is None:
            seasons = ["2024-25"]
        self.seasons = seasons if isinstance(seasons, list) else [seasons]
        self.leagues = [
            "ENG-Premier League",
            "ESP-La Liga", 
            "ITA-Serie A",
            "GER-Bundesliga",
            "FRA-Ligue 1"
        ]
        
        # Create directory structure
        self.setup_directories()
        
        logger.info(f"🚀 Initialized FBref collector for seasons: {', '.join(self.seasons)}")
    
    def setup_directories(self):
        """Create organized directory structure"""
        self.data_dir = Path("data")
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories
        for directory in [self.data_dir, self.raw_dir, self.processed_dir]:
            directory.mkdir(exist_ok=True)
            
        logger.info(f"📁 Directory structure created: {self.data_dir}")
    
    def collect_all_player_data(self) -> Dict[str, pd.DataFrame]:
        """Collect all available FBref player data for all seasons"""
        
        logger.info("📊 Starting comprehensive FBref data collection...")
        all_data = {}
        
        # Define all stat types with correct method and parameters
        stat_types = {
            'standard': ('read_player_season_stats', 'standard'),
            'shooting': ('read_player_season_stats', 'shooting'),
            'passing': ('read_player_season_stats', 'passing'),
            'passing_types': ('read_player_season_stats', 'passing_types'),
            'goal_shot_creation': ('read_player_season_stats', 'goal_shot_creation'),
            'defense': ('read_player_season_stats', 'defense'),
            'possession': ('read_player_season_stats', 'possession'),
            'playing_time': ('read_player_season_stats', 'playing_time'),
            'misc': ('read_player_season_stats', 'misc'),
            'keeper': ('read_player_season_stats', 'keeper'),
            'keeper_adv': ('read_player_season_stats', 'keeper_adv')
        }
        
        # Collect data for each season
        for season in self.seasons:
            logger.info(f"🗓️  Collecting data for season: {season}")
            
            # Initialize FBref scraper for this season
            fbref = sd.FBref(leagues=self.leagues, seasons=season)
            
            for stat_name, (method_name, stat_type) in stat_types.items():
                try:
                    logger.info(f"📈 Collecting {stat_name} stats for {season}...")
                    
                    # Get the method and call it with stat_type parameter
                    method = getattr(fbref, method_name, None)
                    if method:
                        data = method(stat_type=stat_type)
                        
                        if data is not None and not data.empty:
                            # Reset index to get player and team as columns
                            if isinstance(data.index, pd.MultiIndex):
                                data = data.reset_index()
                            
                            # Add season column
                            data['season'] = season
                            
                            # Combine with existing data if it exists
                            key = stat_name
                            if key in all_data:
                                all_data[key] = pd.concat([all_data[key], data], ignore_index=True)
                            else:
                                all_data[key] = data
                            
                            logger.info(f"✅ {stat_name} ({season}): {len(data)} records collected")
                            
                            # Save raw data by season
                            raw_file = self.raw_dir / f"fbref_{stat_name}_{season}.csv"
                            data.to_csv(raw_file, index=False)
                            logger.info(f"💾 Saved raw data: {raw_file}")
                        else:
                            logger.warning(f"⚠️  No data returned for {stat_name} ({season})")
                    else:
                        logger.warning(f"⚠️  Method {method_name} not found")
                    
                    # Rate limiting to be respectful to FBref
                    time.sleep(2)
                    
                except Exception as e:
                    logger.error(f"❌ Error collecting {stat_name} for {season}: {e}")
                    continue
        
        logger.info(f"🎉 Data collection complete! Collected {len(all_data)} stat types across {len(self.seasons)} seasons")
        return all_data
    
    def standardize_data_formats(self, all_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Standardize column names and formats across all datasets"""
        
        logger.info("🔧 Standardizing data formats...")
        standardized_data = {}
        
        # Column mapping for consistency - matches server expectations
        column_mappings = {
            # Basic info columns
            'Player': 'player',
            'player': 'player',
            'level_0': 'player',
            'level_1': 'team',
            'Squad': 'team', 
            'team': 'team',
            'Comp': 'league',
            'league': 'league',
            'Pos': 'position',
            'pos': 'position',
            'position': 'position',
            'Age': 'age',
            'age': 'age',
            'nation': 'nation',
            'born': 'born',
            'season': 'season',
            
            # Multi-level FBRef columns - match server expectations exactly
            'playing_time_mp': 'playing_time_mp',
            'playing_time_starts': 'playing_time_starts', 
            'playing_time_min': 'playing_time_min',
            'playing_time_90s': 'playing_time_90s',
            'performance_gls': 'performance_gls',
            'performance_ast': 'performance_ast',
            'performance_g_a': 'performance_g_a',
            'performance_g_pk': 'performance_g_pk',
            'performance_pk': 'performance_pk',
            'performance_pkatt': 'performance_pkatt',
            'performance_crdy': 'performance_crdy',
            'performance_crdr': 'performance_crdr',
            'expected_xg': 'expected_xg',
            'expected_npxg': 'expected_npxg',
            'expected_xag': 'expected_xag',
            'expected_npxg_xag': 'expected_npxg_xag',
            'progression_prgc': 'progression_prgc',
            'progression_prgp': 'progression_prgp',
            'progression_prgr': 'progression_prgr',
            'per_90_minutes_gls': 'per_90_minutes_gls',
            'per_90_minutes_ast': 'per_90_minutes_ast',
            'per_90_minutes_g_a': 'per_90_minutes_g_a',
            'per_90_minutes_g_pk': 'per_90_minutes_g_pk',
            'per_90_minutes_g_a_pk': 'per_90_minutes_g_a_pk',
            'per_90_minutes_xg': 'per_90_minutes_xg',
            'per_90_minutes_xag': 'per_90_minutes_xag',
            'per_90_minutes_xg_xag': 'per_90_minutes_xg_xag',
            'per_90_minutes_npxg': 'per_90_minutes_npxg',
            'per_90_minutes_npxg_xag': 'per_90_minutes_npxg_xag',
            
            # Shooting stats
            'standard_sh': 'standard_sh',
            'standard_sot': 'standard_sot',
            'standard_sot_pct': 'standard_sot_pct',
            
            # Defensive stats
            'tackles_tkl': 'tackles_tkl',
            'tackles_tklw': 'tackles_tklw',
            'interceptions': 'interceptions',
            'blocks_blocks': 'blocks_blocks',
            'clearances': 'clearances',
            'aerial_duels_won_pct': 'aerial_duels_won_pct',
            
            # Passing stats  
            'total_cmp_pct': 'total_cmp_pct',
            'total_att': 'total_att',
            'progressive_passes': 'progressive_passes',
            'kp': 'kp',
            'carries_prgc': 'carries_prgc',
            'take_ons_succ': 'take-ons_succ',
            'touches_att_pen': 'touches_att_pen',
            'sca_sca': 'sca_sca',
            'gca_gca': 'gca_gca'
        }
        
        for stat_type, data in all_data.items():
            try:
                # Create a copy to work with
                df = data.copy()
                
                # Standardize column names
                new_columns = []
                for col in df.columns:
                    if isinstance(col, tuple):
                        # Handle multi-level column names from FBRef
                        category = str(col[0]).strip() if len(col) > 0 and str(col[0]) != 'nan' and 'Unnamed:' not in str(col[0]) else ''
                        stat = str(col[1]).strip() if len(col) > 1 and str(col[1]) != 'nan' and 'level_' not in str(col[1]) else ''
                        
                        # For basic info columns (first few columns), use the stat name directly
                        basic_info_stats = ['', 'league', 'season', 'team', 'player', 'nation', 'pos', 'age', 'born']
                        if stat in basic_info_stats or category == '' or 'Unnamed:' in category:
                            col_str = stat if stat else category
                        else:
                            # Create meaningful column names for actual stats
                            if category and stat:
                                col_str = f"{category}_{stat}".lower().replace(' ', '_').replace('.', '_').replace('%', '_pct').replace('+', '_').replace('-', '_').replace('(', '').replace(')', '')
                            elif stat:
                                col_str = stat.lower().replace(' ', '_').replace('.', '_').replace('%', '_pct').replace('+', '_').replace('-', '_').replace('(', '').replace(')', '')
                            else:
                                col_str = category.lower().replace(' ', '_').replace('.', '_').replace('%', '_pct').replace('+', '_').replace('-', '_').replace('(', '').replace(')', '')
                    else:
                        col_str = str(col)
                    
                    # Apply mapping or clean the column name
                    clean_col = column_mappings.get(col_str, col_str.lower().replace(' ', '_').replace('.', '_').replace('%', '_pct'))
                    new_columns.append(clean_col)
                
                df.columns = new_columns
                
                # Remove duplicate or problematic columns
                df = df.loc[:, ~df.columns.duplicated()]
                
                # Clean player names (remove extra characters)
                if 'player' in df.columns:
                    df['player'] = df['player'].str.strip()
                    # Remove country codes in parentheses
                    df['player'] = df['player'].str.replace(r'\s+\([^)]*\)', '', regex=True)
                
                # Add stat_type column (season already added during collection)
                df['stat_type'] = stat_type
                
                # Convert numeric columns
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                for col in numeric_columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                standardized_data[stat_type] = df
                logger.info(f"✅ Standardized {stat_type}: {len(df)} rows, {len(df.columns)} columns")
                
            except Exception as e:
                logger.error(f"❌ Error standardizing {stat_type}: {e}")
                continue
        
        return standardized_data
    
    def create_unified_dataset(self, standardized_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create unified player dataset for MCP server"""
        
        logger.info("🔄 Creating unified player dataset...")
        
        # Start with standard stats as base
        if 'standard' not in standardized_data:
            logger.error("❌ No standard stats found - cannot create unified dataset")
            return pd.DataFrame()
        
        unified_df = standardized_data['standard'].copy()
        logger.info(f"📊 Base dataset: {len(unified_df)} players")
        
        # Merge columns for joining
        merge_cols = ['player', 'team', 'league', 'season']
        
        # Merge all other stat types
        for stat_type, data in standardized_data.items():
            if stat_type == 'standard':
                continue
                
            try:
                # Find common columns for merging
                common_cols = [col for col in merge_cols if col in data.columns and col in unified_df.columns]
                
                if common_cols:
                    # Merge on common columns
                    before_count = len(unified_df)
                    unified_df = pd.merge(
                        unified_df, 
                        data, 
                        on=common_cols, 
                        how='left',
                        suffixes=('', f'_{stat_type}')
                    )
                    
                    logger.info(f"✅ Merged {stat_type}: {before_count} → {len(unified_df)} players")
                else:
                    logger.warning(f"⚠️  Cannot merge {stat_type} - no common columns")
                    
            except Exception as e:
                logger.error(f"❌ Error merging {stat_type}: {e}")
                continue
        
        # Clean up the unified dataset
        unified_df = self.clean_unified_dataset(unified_df)
        
        logger.info(f"🎉 Unified dataset complete: {len(unified_df)} players, {len(unified_df.columns)} columns")
        return unified_df
    
    def clean_unified_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and optimize the unified dataset"""
        
        logger.info("🧹 Cleaning unified dataset...")
        
        # Remove players with minimal playing time (less than 90 minutes total)
        if 'minutes_played' in df.columns:
            before_count = len(df)
            df = df[df['minutes_played'].fillna(0) >= 90]
            logger.info(f"📊 Filtered by playing time: {before_count} → {len(df)} players")
        
        # Fill missing values intelligently
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        # Fill missing categorical values
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df[col] = df[col].fillna('Unknown')
        
        # Remove completely empty columns
        df = df.dropna(axis=1, how='all')
        
        # Sort by league and team for better organization
        if 'league' in df.columns and 'team' in df.columns:
            df = df.sort_values(['league', 'team', 'player'])
        
        return df
    
    def create_mcp_optimized_files(self, unified_df: pd.DataFrame):
        """Create files optimized for MCP server consumption"""
        
        logger.info("📄 Creating MCP-optimized files...")
        
        # Main unified file for MCP server
        main_file = self.data_dir / "unified_player_stats.csv"
        unified_df.to_csv(main_file, index=False)
        logger.info(f"💾 Main file saved: {main_file} ({len(unified_df)} players)")
        
        # Create separate files by league for faster queries
        if 'league' in unified_df.columns:
            for league in unified_df['league'].unique():
                if pd.notna(league) and league != 'Unknown':
                    league_df = unified_df[unified_df['league'] == league]
                    safe_league = league.replace(' ', '_').replace('-', '_')
                    league_file = self.data_dir / f"{safe_league.lower()}_players.csv"
                    league_df.to_csv(league_file, index=False)
                    logger.info(f"📊 {league}: {len(league_df)} players → {league_file}")
        
        # Create position-based files for faster queries  
        if 'position' in unified_df.columns:
            for position in unified_df['position'].unique():
                if pd.notna(position) and position != 'Unknown':
                    pos_df = unified_df[unified_df['position'].str.contains(position, case=False, na=False)]
                    if len(pos_df) > 10:  # Only create if substantial data
                        safe_pos = position.replace(' ', '_').replace(',', '_').replace('-', '_')
                        pos_file = self.data_dir / f"{safe_pos.lower()}_players.csv"
                        pos_df.to_csv(pos_file, index=False)
                        logger.info(f"⚽ {position}: {len(pos_df)} players → {pos_file}")
        
        # Create summary file for quick reference
        summary_data = {
            'total_players': len(unified_df),
            'leagues': unified_df['league'].nunique() if 'league' in unified_df.columns else 0,
            'teams': unified_df['team'].nunique() if 'team' in unified_df.columns else 0,
            'positions': unified_df['position'].nunique() if 'position' in unified_df.columns else 0,
            'columns': len(unified_df.columns),
            'seasons': self.seasons,
            'created': datetime.now().isoformat()
        }
        
        summary_file = self.data_dir / "data_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        logger.info(f"📋 Summary saved: {summary_file}")
    
    def run_complete_collection(self):
        """Run the complete data collection pipeline"""
        
        logger.info("🚀 Starting complete FBref data collection pipeline...")
        start_time = time.time()
        
        try:
            # Step 1: Collect all raw data
            all_data = self.collect_all_player_data()
            
            if not all_data:
                logger.error("❌ No data collected - aborting pipeline")
                return
            
            # Step 2: Standardize formats
            standardized_data = self.standardize_data_formats(all_data)
            
            # Step 3: Create unified dataset
            unified_df = self.create_unified_dataset(standardized_data)
            
            if unified_df.empty:
                logger.error("❌ Failed to create unified dataset")
                return
            
            # Step 4: Create MCP-optimized files
            self.create_mcp_optimized_files(unified_df)
            
            # Final summary
            elapsed_time = time.time() - start_time
            logger.info("=" * 60)
            logger.info("🎉 DATA COLLECTION COMPLETE!")
            logger.info(f"⏱️  Time taken: {elapsed_time:.1f} seconds")
            logger.info(f"📊 Players collected: {len(unified_df)}")
            logger.info(f"📈 Metrics per player: {len(unified_df.columns)}")
            logger.info(f"📁 Files saved to: {self.data_dir}")
            logger.info("=" * 60)
            
            # Display sample data
            logger.info("📋 Sample of collected data:")
            if 'player' in unified_df.columns:
                sample = unified_df[['player', 'team', 'league', 'position']].head()
                logger.info(f"\n{sample.to_string(index=False)}")
            
            return unified_df
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise

# Main execution function
def main():
    """Main function to run the FBref data collection"""
    
    print("🚀 FBref Comprehensive Data Collector")
    print("=====================================")
    
    # Get seasons from command line arguments or use defaults
    import sys
    if len(sys.argv) > 1:
        seasons = sys.argv[1:]
        print(f"📅 Collecting data for seasons: {', '.join(seasons)}")
    else:
        seasons = ["2023-24", "2024-25", "2025-26"]
        print(f"📅 No seasons specified, using defaults: {', '.join(seasons)}")
    
    try:
        # Initialize collector with multiple seasons
        collector = ComprehensiveFBrefCollector(seasons=seasons)
        
        # Run complete collection pipeline
        result = collector.run_complete_collection()
        
        if result is not None and not result.empty:
            print("\n✅ SUCCESS! Your comprehensive FBref data is ready for the MCP server.")
            print(f"📁 Data location: {collector.data_dir}")
            print(f"📊 Total seasons collected: {len(seasons)}")
            print("🔧 Next step: Run your MCP server with this data!")
        else:
            print("\n❌ FAILED! Could not collect data.")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("Please check your internet connection and try again.")

if __name__ == "__main__":
    main()