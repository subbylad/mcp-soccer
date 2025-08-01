#!/usr/bin/env python3
"""
Enhanced Soccer Data MCP Server
Professional-grade scouting and analysis system
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
from typing import Optional, List, Dict, Any, Union
import logging
from dataclasses import dataclass
from enum import Enum
import re

# Configure logging to stderr only
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Data Models
@dataclass
class PlayerBasicInfo:
    name: str
    age: int
    nationality: str
    team: str
    league: str
    position: str
    secondary_positions: List[str]
    season: str

@dataclass
class PerformanceStats:
    minutes_played: float
    games_played: int
    goals: float
    assists: float
    expected_goals: float
    expected_assists: float
    goals_per_90: float
    assists_per_90: float
    
@dataclass
class DefensiveStats:
    tackles_per_90: float
    tackle_success_rate: float
    interceptions_per_90: float
    clearances: float
    aerial_duels_won_pct: float
    blocks: float
    pressures: float

@dataclass
class PassingStats:
    pass_completion_pct: float
    progressive_passes: float
    key_passes: float
    long_passes_completed: float
    crosses_accuracy: float
    progressive_carries: float

@dataclass
class AttackingStats:
    shots_per_90: float
    shots_on_target_pct: float
    big_chances_created: float
    dribbles_attempted: float
    dribble_success_rate: float
    touches_penalty_area: float

@dataclass
class PlayerSummary:
    name: str
    age: int
    team: str
    league: str
    position: str
    key_stats: Dict[str, float]
    overall_rating: float

class ScoutingPosition(Enum):
    GK = "GK"
    DF = "DF" 
    MF = "MF"
    FW = "FW"
    
class EnhancedSoccerDataServer:
    def __init__(self):
        self.data = self.load_comprehensive_data()
        self.position_mappings = self._create_position_mappings()
        self.stat_mappings = self._create_stat_mappings()
        logger.info(f"Enhanced server loaded {len(self.data)} players with {len(self.data.columns)} metrics")
    
    def load_comprehensive_data(self):
        """Load and enhance the unified player stats data"""
        script_dir = Path(__file__).parent
        data_file = script_dir / "data" / "unified_player_stats.csv"
        
        if not data_file.exists():
            logger.error(f"Data file not found: {data_file}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(data_file)
            
            # Data quality improvements
            df = self._enhance_data_quality(df)
            
            logger.info(f"Loaded comprehensive dataset: {len(df)} players, {len(df.columns)} columns")
            
            # Log available leagues and positions for debugging
            if 'league' in df.columns:
                unique_leagues = df['league'].unique()
                logger.info(f"Available leagues: {list(unique_leagues)}")
                
            if 'position' in df.columns:
                unique_positions = df['position'].dropna().unique()
                logger.info(f"Available positions: {list(unique_positions)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def _enhance_data_quality(self, df):
        """Enhance data quality and add computed metrics"""
        
        # Clean position data
        if 'position' in df.columns:
            df['position'] = df['position'].fillna('Unknown')
            # Split multi-positions and create primary position
            df['primary_position'] = df['position'].apply(self._extract_primary_position)
            df['secondary_positions'] = df['position'].apply(self._extract_secondary_positions)
        
        # Ensure numeric columns are properly typed
        numeric_columns = [
            'age', 'playing_time_mp', 'performance_gls', 'performance_ast',
            'expected_xg', 'expected_npxg', 'standard_sh', 'standard_sot',
            'tackles_tkl', 'interceptions', 'aerial_duels_won_pct'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Add computed per-90 metrics where missing
        if 'playing_time_90s' in df.columns:
            minutes_90s = df['playing_time_90s']
            if 'tackles_per_90' not in df.columns and 'tackles_tkl' in df.columns:
                df['tackles_per_90'] = (df['tackles_tkl'] / minutes_90s).round(2)
            
            if 'interceptions_per_90' not in df.columns and 'interceptions' in df.columns:
                df['interceptions_per_90'] = (df['interceptions'] / minutes_90s).round(2)
        
        return df
    
    def _extract_primary_position(self, position_str):
        """Extract primary position from multi-position strings"""
        if pd.isna(position_str) or position_str == 'Unknown':
            return 'Unknown'
        
        positions = str(position_str).split(',')
        return positions[0].strip()
    
    def _extract_secondary_positions(self, position_str):
        """Extract secondary positions as list"""
        if pd.isna(position_str) or position_str == 'Unknown':
            return []
        
        positions = str(position_str).split(',')
        return [pos.strip() for pos in positions[1:]] if len(positions) > 1 else []
    
    def _create_position_mappings(self):
        """Create mappings for position filtering"""
        return {
            'goalkeeper': ['GK'],
            'defender': ['DF', 'DF,MF', 'MF,DF'],
            'midfielder': ['MF', 'MF,DF', 'MF,FW', 'DF,MF', 'FW,MF'],
            'forward': ['FW', 'FW,MF', 'MF,FW'],
            'centre_back': ['DF'],
            'full_back': ['DF'],
            'defensive_midfielder': ['MF', 'DF,MF'],
            'central_midfielder': ['MF'],
            'attacking_midfielder': ['MF', 'MF,FW'],
            'winger': ['FW,MF', 'MF,FW'],
            'striker': ['FW']
        }
    
    def _create_stat_mappings(self):
        """Map friendly stat names to column names"""
        return {
            'goals': 'performance_gls',
            'assists': 'performance_ast',
            'minutes_played': 'playing_time_min',
            'games_played': 'playing_time_mp',
            'expected_goals': 'expected_xg',
            'expected_assists': 'expected_xag',
            'shots': 'standard_sh',
            'shots_on_target': 'standard_sot',
            'shots_on_target_pct': 'standard_sot_pct',
            'pass_completion_pct': 'total_cmp_pct',
            'tackles': 'tackles_tkl',
            'tackles_per_90': 'tackles_per_90',
            'interceptions': 'interceptions',
            'interceptions_per_90': 'interceptions_per_90',
            'aerial_duels_won_pct': 'aerial_duels_won_pct',
            'progressive_passes': 'progressive_passes',
            'progressive_carries': 'carries_prgc',
            'successful_dribbles': 'take-ons_succ',
            'penalty_area_touches': 'touches_att_pen'
        }
    
    # ENHANCED SEARCH FUNCTIONS
    def search_players_advanced(
        self,
        leagues: Optional[List[str]] = None,
        positions: Optional[List[str]] = None,
        age_min: Optional[int] = None,
        age_max: Optional[int] = None,
        nationality: Optional[List[str]] = None,
        team: Optional[str] = None,
        min_minutes_played: Optional[int] = 500,
        stat_filters: Optional[Dict[str, float]] = None,
        limit: int = 100,
        sort_by: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Advanced player search with comprehensive filtering"""
        
        filtered = self.data.copy()
        
        # Apply filters
        if leagues:
            league_filter = filtered['league'].isin(leagues)
            filtered = filtered[league_filter]
            
        if positions:
            # Support both exact position matches and positional roles
            position_conditions = []
            for pos in positions:
                if pos in self.position_mappings:
                    # Positional role mapping
                    mapped_positions = self.position_mappings[pos]
                    pos_condition = filtered['position'].isin(mapped_positions)
                else:
                    # Direct position match
                    pos_condition = filtered['position'].str.contains(pos, case=False, na=False)
                position_conditions.append(pos_condition)
            
            if position_conditions:
                combined_position_filter = position_conditions[0]
                for condition in position_conditions[1:]:
                    combined_position_filter |= condition
                filtered = filtered[combined_position_filter]
        
        if age_min is not None:
            filtered = filtered[filtered['age'] >= age_min]
            
        if age_max is not None:
            filtered = filtered[filtered['age'] <= age_max]
            
        if nationality:
            filtered = filtered[filtered['nation'].isin(nationality)]
            
        if team:
            filtered = filtered[filtered['team'].str.contains(team, case=False, na=False)]
            
        if min_minutes_played:
            minutes_col = 'playing_time_min' if 'playing_time_min' in filtered.columns else 'playing_time_mp'
            if minutes_col in filtered.columns:
                filtered = filtered[filtered[minutes_col] >= min_minutes_played]
        
        # Apply statistical filters
        if stat_filters:
            for stat_name, min_value in stat_filters.items():
                column_name = self.stat_mappings.get(stat_name, stat_name)
                if column_name in filtered.columns:
                    filtered = filtered[filtered[column_name] >= min_value]
        
        # Sorting
        if sort_by:
            sort_column = self.stat_mappings.get(sort_by, sort_by)
            if sort_column in filtered.columns:
                filtered = filtered.sort_values(sort_column, ascending=False)
        
        # Limit results
        result = filtered.head(limit)
        
        # Format results
        players = []
        for _, player in result.iterrows():
            player_summary = {
                'name': player.get('player', 'Unknown'),
                'age': int(player.get('age', 0)) if pd.notna(player.get('age')) else 0,
                'team': player.get('team', 'Unknown'),
                'league': player.get('league', 'Unknown'),
                'position': player.get('position', 'Unknown'),
                'nationality': player.get('nation', 'Unknown'),
                'key_stats': {
                    'goals': float(player.get('performance_gls', 0)) if pd.notna(player.get('performance_gls')) else 0,
                    'assists': float(player.get('performance_ast', 0)) if pd.notna(player.get('performance_ast')) else 0,
                    'minutes_played': float(player.get('playing_time_min', 0)) if pd.notna(player.get('playing_time_min')) else 0,
                    'expected_goals': float(player.get('expected_xg', 0)) if pd.notna(player.get('expected_xg')) else 0
                }
            }
            players.append(player_summary)
        
        return players
    
    def search_by_profile(self, scout_brief: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search players based on comprehensive scouting profile"""
        
        # Extract search parameters from scout brief
        leagues = scout_brief.get('target_leagues', None)
        positions = scout_brief.get('positions', None)
        age_range = scout_brief.get('age_range', {})
        physical_requirements = scout_brief.get('physical_requirements', {})
        technical_requirements = scout_brief.get('technical_requirements', {})
        
        # Build statistical filters
        stat_filters = {}
        
        # Add technical requirements
        if technical_requirements:
            for req, min_val in technical_requirements.items():
                stat_filters[req] = min_val
        
        # Perform advanced search
        return self.search_players_advanced(
            leagues=leagues,
            positions=positions,
            age_min=age_range.get('min'),
            age_max=age_range.get('max'),
            stat_filters=stat_filters,
            limit=scout_brief.get('max_results', 50)
        )
    
    def get_league_leaders(
        self,
        league: str,
        stat: str,
        position: Optional[str] = None,
        min_games: int = 15
    ) -> List[Dict[str, Any]]:
        """Get league leaders in specific statistics"""
        
        filtered = self.data[self.data['league'] == league].copy()
        
        if position:
            filtered = filtered[filtered['position'].str.contains(position, case=False, na=False)]
        
        # Filter by minimum games
        games_col = 'playing_time_mp' if 'playing_time_mp' in filtered.columns else 'playing_time_starts'
        if games_col in filtered.columns:
            filtered = filtered[filtered[games_col] >= min_games]
        
        # Get stat column
        stat_column = self.stat_mappings.get(stat, stat)
        if stat_column not in filtered.columns:
            return []
        
        # Sort and get top performers
        top_performers = filtered.nlargest(20, stat_column)
        
        leaders = []
        for _, player in top_performers.iterrows():
            leader_info = {
                'name': player.get('player', 'Unknown'),
                'team': player.get('team', 'Unknown'),
                'position': player.get('position', 'Unknown'),
                'stat_value': float(player.get(stat_column, 0)) if pd.notna(player.get(stat_column)) else 0,
                'games_played': int(player.get(games_col, 0)) if pd.notna(player.get(games_col)) else 0
            }
            leaders.append(leader_info)
        
        return leaders
    
    def compare_multiple_players(
        self,
        player_names: List[str],
        focus_stats: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compare multiple players across key statistics"""
        
        if not focus_stats:
            focus_stats = [
                'goals', 'assists', 'expected_goals', 'shots_on_target_pct',
                'pass_completion_pct', 'tackles_per_90', 'progressive_passes'
            ]
        
        comparison_data = {}
        players_found = []
        
        for player_name in player_names:
            matches = self.data[self.data['player'].str.contains(player_name, case=False, na=False)]
            
            if len(matches) > 0:
                player = matches.iloc[0]
                players_found.append(player_name)
                
                player_stats = {
                    'name': player.get('player', 'Unknown'),
                    'age': int(player.get('age', 0)) if pd.notna(player.get('age')) else 0,
                    'team': player.get('team', 'Unknown'),
                    'league': player.get('league', 'Unknown'),
                    'position': player.get('position', 'Unknown'),
                    'stats': {}
                }
                
                # Extract focus statistics
                for stat in focus_stats:
                    column_name = self.stat_mappings.get(stat, stat)
                    if column_name in player.index:
                        stat_value = player.get(column_name, 0)
                        player_stats['stats'][stat] = float(stat_value) if pd.notna(stat_value) else 0
                    else:
                        player_stats['stats'][stat] = 0
                
                comparison_data[player_name] = player_stats
        
        return {
            'players_compared': len(players_found),
            'players_found': players_found,
            'players_not_found': [name for name in player_names if name not in players_found],
            'comparison_data': comparison_data,
            'focus_statistics': focus_stats
        }
    
    def generate_detailed_scouting_report(
        self,
        player_name: str,
        comparison_players: Optional[List[str]] = None,
        focus_areas: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive scouting report for a player"""
        
        # Find the player
        matches = self.data[self.data['player'].str.contains(player_name, case=False, na=False)]
        
        if len(matches) == 0:
            return {'error': f"Player '{player_name}' not found"}
        
        if len(matches) > 1:
            return {
                'error': f"Multiple players found for '{player_name}'",
                'options': [f"{row['player']} ({row['team']})" for _, row in matches.head(5).iterrows()]
            }
        
        player = matches.iloc[0]
        
        # Basic player information
        basic_info = {
            'name': player.get('player', 'Unknown'),
            'age': int(player.get('age', 0)) if pd.notna(player.get('age')) else 0,
            'nationality': player.get('nation', 'Unknown'),
            'team': player.get('team', 'Unknown'),
            'league': player.get('league', 'Unknown'),
            'position': player.get('position', 'Unknown'),
            'season': player.get('season', '2024-25')
        }
        
        # Performance statistics organized by category
        performance_stats = self._extract_performance_stats(player)
        defensive_stats = self._extract_defensive_stats(player)
        passing_stats = self._extract_passing_stats(player)
        attacking_stats = self._extract_attacking_stats(player)
        
        # Generate comparison if requested
        comparison_data = None
        if comparison_players:
            comparison_data = self.compare_multiple_players(
                [player_name] + comparison_players,
                focus_stats=['goals', 'assists', 'expected_goals', 'tackles_per_90', 'pass_completion_pct']
            )
        
        return {
            'basic_info': basic_info,
            'performance_stats': performance_stats,
            'defensive_stats': defensive_stats,
            'passing_stats': passing_stats,
            'attacking_stats': attacking_stats,
            'comparison_data': comparison_data,
            'scouting_summary': self._generate_scouting_summary(player, focus_areas)
        }
    
    def _extract_performance_stats(self, player) -> Dict[str, float]:
        """Extract core performance statistics"""
        return {
            'goals': float(player.get('performance_gls', 0)) if pd.notna(player.get('performance_gls')) else 0,
            'assists': float(player.get('performance_ast', 0)) if pd.notna(player.get('performance_ast')) else 0,
            'minutes_played': float(player.get('playing_time_min', 0)) if pd.notna(player.get('playing_time_min')) else 0,
            'games_played': int(player.get('playing_time_mp', 0)) if pd.notna(player.get('playing_time_mp')) else 0,
            'expected_goals': float(player.get('expected_xg', 0)) if pd.notna(player.get('expected_xg')) else 0,
            'expected_assists': float(player.get('expected_xag', 0)) if pd.notna(player.get('expected_xag')) else 0,
            'goals_per_90': float(player.get('per_90_minutes_gls', 0)) if pd.notna(player.get('per_90_minutes_gls')) else 0,
            'assists_per_90': float(player.get('per_90_minutes_ast', 0)) if pd.notna(player.get('per_90_minutes_ast')) else 0
        }
    
    def _extract_defensive_stats(self, player) -> Dict[str, float]:
        """Extract defensive statistics"""
        return {
            'tackles': float(player.get('tackles_tkl', 0)) if pd.notna(player.get('tackles_tkl')) else 0,
            'tackles_won': float(player.get('tackles_tklw', 0)) if pd.notna(player.get('tackles_tklw')) else 0,
            'interceptions': float(player.get('interceptions', 0)) if pd.notna(player.get('interceptions')) else 0,
            'blocks': float(player.get('blocks_blocks', 0)) if pd.notna(player.get('blocks_blocks')) else 0,
            'clearances': float(player.get('clearances', 0)) if pd.notna(player.get('clearances')) else 0,
            'aerial_duels_won': float(player.get('aerial_duels_won', 0)) if pd.notna(player.get('aerial_duels_won')) else 0,
            'aerial_duels_won_pct': float(player.get('aerial_duels_won_pct', 0)) if pd.notna(player.get('aerial_duels_won_pct')) else 0
        }
    
    def _extract_passing_stats(self, player) -> Dict[str, float]:
        """Extract passing and progression statistics"""
        return {
            'pass_completion_pct': float(player.get('total_cmp_pct', 0)) if pd.notna(player.get('total_cmp_pct')) else 0,
            'passes_attempted': float(player.get('total_att', 0)) if pd.notna(player.get('total_att')) else 0,
            'progressive_passes': float(player.get('progressive_passes', 0)) if pd.notna(player.get('progressive_passes')) else 0,
            'key_passes': float(player.get('kp', 0)) if pd.notna(player.get('kp')) else 0,
            'long_passes_completed': float(player.get('long_cmp', 0)) if pd.notna(player.get('long_cmp')) else 0,
            'progressive_carries': float(player.get('carries_prgc', 0)) if pd.notna(player.get('carries_prgc')) else 0
        }
    
    def _extract_attacking_stats(self, player) -> Dict[str, float]:
        """Extract attacking statistics"""
        return {
            'shots': float(player.get('standard_sh', 0)) if pd.notna(player.get('standard_sh')) else 0,
            'shots_on_target': float(player.get('standard_sot', 0)) if pd.notna(player.get('standard_sot')) else 0,
            'shots_on_target_pct': float(player.get('standard_sot_pct', 0)) if pd.notna(player.get('standard_sot_pct')) else 0,
            'successful_dribbles': float(player.get('take-ons_succ', 0)) if pd.notna(player.get('take-ons_succ')) else 0,
            'dribble_attempts': float(player.get('take-ons_att', 0)) if pd.notna(player.get('take-ons_att')) else 0,
            'penalty_area_touches': float(player.get('touches_att_pen', 0)) if pd.notna(player.get('touches_att_pen')) else 0,
            'shot_creating_actions': float(player.get('sca_sca', 0)) if pd.notna(player.get('sca_sca')) else 0,
            'goal_creating_actions': float(player.get('gca_gca', 0)) if pd.notna(player.get('gca_gca')) else 0
        }
    
    def _generate_scouting_summary(self, player, focus_areas: Optional[List[str]] = None) -> Dict[str, str]:
        """Generate textual scouting summary"""
        
        summary = {}
        
        # Overall assessment
        goals = float(player.get('performance_gls', 0)) if pd.notna(player.get('performance_gls')) else 0
        assists = float(player.get('performance_ast', 0)) if pd.notna(player.get('performance_ast')) else 0
        age = int(player.get('age', 0)) if pd.notna(player.get('age')) else 0
        
        summary['overall'] = f"A {age}-year-old player with {goals} goals and {assists} assists this season."
        
        # Position-specific insights
        position = player.get('position', 'Unknown')
        if 'FW' in str(position):
            xg = float(player.get('expected_xg', 0)) if pd.notna(player.get('expected_xg')) else 0
            summary['attacking'] = f"As a forward, showing clinical finishing with {goals} goals from {xg:.1f} expected goals."
        
        elif 'MF' in str(position):
            passes = float(player.get('total_cmp_pct', 0)) if pd.notna(player.get('total_cmp_pct')) else 0
            summary['passing'] = f"Midfielder with {passes:.1f}% pass completion rate, contributing {assists} assists."
        
        elif 'DF' in str(position):
            tackles = float(player.get('tackles_tkl', 0)) if pd.notna(player.get('tackles_tkl')) else 0
            summary['defensive'] = f"Defender with solid defensive stats including {tackles} tackles."
        
        return summary

# Initialize the enhanced server
server = EnhancedSoccerDataServer()

# MCP Protocol Handler with enhanced functions
def handle_mcp_request(request):
    """Handle MCP requests for enhanced soccer server"""
    try:
        method = request.get('method')
        request_id = request.get('id')
        
        if method == 'initialize':
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": "enhanced-soccer-data",
                        "version": "2.0.0"
                    }
                }
            }
        
        elif method == 'notifications/initialized':
            return None
            
        elif method == 'tools/list':
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "tools": [
                        {
                            "name": "search_players_advanced",
                            "description": "Advanced player search with comprehensive filtering (leagues, positions, age range, stats)",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "leagues": {"type": "array", "items": {"type": "string"}, "description": "Target leagues (e.g., ['ENG-Premier League', 'ESP-La Liga'])"},
                                    "positions": {"type": "array", "items": {"type": "string"}, "description": "Positions or roles (e.g., ['MF', 'defensive_midfielder'])"},
                                    "age_min": {"type": "integer", "description": "Minimum age"},
                                    "age_max": {"type": "integer", "description": "Maximum age"}, 
                                    "nationality": {"type": "array", "items": {"type": "string"}, "description": "Player nationalities"},
                                    "team": {"type": "string", "description": "Team name"},
                                    "min_minutes_played": {"type": "integer", "description": "Minimum minutes played"},
                                    "stat_filters": {"type": "object", "description": "Statistical requirements (e.g., {'tackles_per_90': 1.5})"},
                                    "limit": {"type": "integer", "default": 50, "description": "Maximum results"},
                                    "sort_by": {"type": "string", "description": "Sort by statistic"}
                                }
                            }
                        },
                        {
                            "name": "search_by_profile",
                            "description": "Search players based on comprehensive scouting profile",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "scout_brief": {
                                        "type": "object",
                                        "description": "Scouting requirements including target_leagues, positions, age_range, technical_requirements",
                                        "properties": {
                                            "target_leagues": {"type": "array", "items": {"type": "string"}},
                                            "positions": {"type": "array", "items": {"type": "string"}},
                                            "age_range": {"type": "object", "properties": {"min": {"type": "integer"}, "max": {"type": "integer"}}},
                                            "technical_requirements": {"type": "object"},
                                            "max_results": {"type": "integer", "default": 50}
                                        }
                                    }
                                },
                                "required": ["scout_brief"]
                            }
                        },
                        {
                            "name": "get_league_leaders",
                            "description": "Get top performers in a league for specific statistics",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "league": {"type": "string", "description": "League name"},
                                    "stat": {"type": "string", "description": "Statistic to rank by"},
                                    "position": {"type": "string", "description": "Filter by position"},
                                    "min_games": {"type": "integer", "default": 15, "description": "Minimum games played"}
                                },
                                "required": ["league", "stat"]
                            }
                        },
                        {
                            "name": "compare_multiple_players",
                            "description": "Compare multiple players across key statistics",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "player_names": {"type": "array", "items": {"type": "string"}, "description": "List of player names"},
                                    "focus_stats": {"type": "array", "items": {"type": "string"}, "description": "Statistics to focus on"}
                                },
                                "required": ["player_names"]
                            }
                        },
                        {
                            "name": "generate_detailed_scouting_report",
                            "description": "Generate comprehensive scouting report for a player",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "player_name": {"type": "string", "description": "Player name"},
                                    "comparison_players": {"type": "array", "items": {"type": "string"}, "description": "Players to compare against"},
                                    "focus_areas": {"type": "array", "items": {"type": "string"}, "description": "Areas to focus analysis on"}
                                },
                                "required": ["player_name"]
                            }
                        },
                        # Legacy functions for backward compatibility
                        {
                            "name": "search_players",
                            "description": "Basic player search (legacy)",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string", "description": "Player name"},
                                    "league": {"type": "string", "description": "League filter"},
                                    "position": {"type": "string", "description": "Position filter"},
                                    "limit": {"type": "integer", "default": 10}
                                }
                            }
                        },
                        {
                            "name": "get_player_details",
                            "description": "Get detailed player statistics (legacy)",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "player_name": {"type": "string", "description": "Player name"}
                                },
                                "required": ["player_name"]
                            }
                        }
                    ]
                }
            }
        
        elif method == 'resources/list':
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"resources": []}
            }
            
        elif method == 'tools/call':
            tool_name = request['params']['name']
            args = request['params'].get('arguments', {})
            
            # Route to appropriate function
            if tool_name == 'search_players_advanced':
                result = server.search_players_advanced(**args)
            elif tool_name == 'search_by_profile':
                result = server.search_by_profile(**args)
            elif tool_name == 'get_league_leaders':
                result = server.get_league_leaders(**args)
            elif tool_name == 'compare_multiple_players':
                result = server.compare_multiple_players(**args)
            elif tool_name == 'generate_detailed_scouting_report':
                result = server.generate_detailed_scouting_report(**args)
            # Legacy functions
            elif tool_name == 'search_players':
                # Convert to advanced search
                result = server.search_players_advanced(
                    leagues=[args.get('league')] if args.get('league') else None,
                    positions=[args.get('position')] if args.get('position') else None,
                    limit=args.get('limit', 10)
                )
                # Format for legacy compatibility
                formatted_result = f"Found {len(result)} players:\\n\\n"
                for player in result:
                    formatted_result += f"🏃 {player['name']}\\n"
                    formatted_result += f"   Team: {player['team']}\\n"
                    formatted_result += f"   League: {player['league']}\\n"
                    formatted_result += f"   Position: {player['position']}\\n"
                    formatted_result += f"   Age: {player['age']}\\n\\n"
                result = formatted_result
            elif tool_name == 'get_player_details':
                scouting_report = server.generate_detailed_scouting_report(args['player_name'])
                if 'error' in scouting_report:
                    result = scouting_report['error']
                else:
                    # Format for legacy compatibility
                    basic = scouting_report['basic_info']
                    perf = scouting_report['performance_stats']
                    result = f"📊 Detailed Stats for {basic['name']}\\n\\n"
                    result += f"Team: {basic['team']}\\n"
                    result += f"League: {basic['league']}\\n"
                    result += f"Position: {basic['position']}\\n"
                    result += f"Age: {basic['age']}\\n\\n"
                    result += "📈 Key Statistics:\\n"
                    for stat, value in perf.items():
                        result += f"   {stat.replace('_', ' ').title()}: {value}\\n"
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}
                }
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result, indent=2) if isinstance(result, (dict, list)) else str(result)}]
                }
            }
        
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"}
            }
            
    except Exception as e:
        logger.error(f"Error handling request: {e}")
        return {
            "jsonrpc": "2.0",
            "id": request.get('id'),
            "error": {"code": -32603, "message": str(e)}
        }

# Main loop
if __name__ == "__main__":
    logger.info("Enhanced Soccer Data MCP Server Started")
    logger.info("Professional scouting capabilities enabled")
    
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            
            try:
                request = json.loads(line)
                response = handle_mcp_request(request)
                
                if response is not None:
                    print(json.dumps(response), flush=True)
                    
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON received: {e}")
                continue
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                continue
                
    except KeyboardInterrupt:
        logger.info("Enhanced Soccer Data MCP Server Stopped")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")