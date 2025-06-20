import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from sentence_transformers import SentenceTransformer
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SoccerDataProcessor:
    """Process soccer dataset to create comprehensive player profiles"""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.dataframes = {}
        self.league_mappings = self.initialize_league_mappings()
        self.competition_lookup = {}  # Will be populated after loading competitions

    def initialize_league_mappings(self) -> dict:
        """Initialize comprehensive league mappings for RAG system"""
        return {
            # Top 5 European leagues - using actual competition IDs from your data
            'premier league': ['GB1', 'premier-league', 'england-first-division', 'premier league'],
            'la liga': ['ES1', 'laliga', 'primera-division', 'spain-primera-division', 'la liga'],
            'bundesliga': ['L1', 'bundesliga', 'germany-first-division', '1-bundesliga'],
            'serie a': ['IT1', 'serie-a', 'italy-primera-division', 'serie a'],
            'ligue 1': ['FR1', 'ligue-1', 'france-primera-division', 'ligue 1'],
            
            # Other major leagues - using actual IDs
            'eredivisie': ['NL1', 'eredivisie', 'netherlands-primera-division'],
            'primeira liga': ['PO1', 'liga-portugal-bwin', 'portugal-primera-division'],
            'super lig': ['TR1', 'super-lig', 'turkey-first-division'],
            'premier liga ukraine': ['UKR1', 'premier-liga', 'ukraine-first-division'],
            'super league greece': ['GR1', 'super-league-1', 'greece-first-division'],
            'scottish premiership': ['SC1', 'scottish-premiership', 'scotland-first-division'],
            'belgian pro league': ['BE1', 'jupiler-pro-league', 'belgium-first-division'],
            'danish superliga': ['DK1', 'superligaen', 'denmark-first-division'],
            'russian premier league': ['RU1', 'premier-liga', 'russia-primera-division'],
            
            # European competitions
            'champions league': ['CL', 'uefa-champions-league', 'champions-league'],
            'europa league': ['EL', 'uefa-europa-league', 'europa-league'],
            'conference league': ['UCOL', 'uefa-conference-league', 'conference-league']
        }

    def clean_market_value(self, value):
        """Clean and standardize market value data"""
        if pd.isna(value) or value == '' or str(value).lower() == 'nan':
            return 0
        
        try:
            # Convert to string and clean
            value_str = str(value).strip()
            
            # Remove currency symbols and common prefixes
            value_str = re.sub(r'[€$£¥]', '', value_str)
            value_str = re.sub(r'[+\-±]', '', value_str)
            value_str = value_str.replace(',', '')
            
            # Handle million/thousand abbreviations
            if 'm' in value_str.lower():
                value_str = value_str.lower().replace('m', '')
                multiplier = 1_000_000
            elif 'k' in value_str.lower():
                value_str = value_str.lower().replace('k', '')
                multiplier = 1_000
            else:
                multiplier = 1
            
            # Extract numeric value
            numeric_match = re.search(r'(\d+(?:\.\d+)?)', value_str)
            if numeric_match:
                numeric_value = float(numeric_match.group(1))
                final_value = numeric_value * multiplier
                
                # Validate reasonable range for market values
                if 0 <= final_value <= 500_000_000:
                    return int(final_value)
            
            return 0
            
        except (ValueError, TypeError):
            return 0

    def clean_club_name(self, name):
        """Clean and standardize club names"""
        if pd.isna(name) or name == '' or str(name).lower() == 'nan':
            return ''
        
        name_str = str(name).strip()
        
        # Remove common suffixes that might cause issues
        suffixes_to_clean = [
            r'\s+\(.*?\)$',  # Remove parenthetical info
            r'\s+FC$', r'\s+F\.C\.$',  # Standardize FC
            r'\s+CF$', r'\s+C\.F\.$',  # Standardize CF
            r'\s+SC$', r'\s+S\.C\.$',  # Standardize SC
        ]
        
        for suffix in suffixes_to_clean:
            name_str = re.sub(suffix, '', name_str, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        name_str = ' '.join(name_str.split())
        
        return name_str

    def create_competition_lookup(self):
        """Create competition ID to name lookup from competitions data"""
        if 'competitions' in self.dataframes:
            self.competition_lookup = {}
            for _, row in self.dataframes['competitions'].iterrows():
                comp_id = row.get('competition_id', '')
                comp_name = row.get('name', '')
                comp_code = row.get('competition_code', '')
                
                if comp_id:
                    self.competition_lookup[comp_id] = {
                        'name': comp_name,
                        'code': comp_code,
                        'type': row.get('type', ''),
                        'country': row.get('country_name', ''),
                        'is_major': row.get('is_major_national_league', False)
                    }
            
            logger.info(f"Created lookup for {len(self.competition_lookup)} competitions")
        
    def load_datasets(self):
        """Load all CSV files into dataframes"""
        csv_files = {
            'players': 'players.csv',
            'appearances': 'appearances.csv', 
            'clubs': 'clubs.csv',
            'competitions': 'competitions.csv',
            'games': 'games.csv',
            'transfers': 'transfers.csv',
            'player_valuations': 'player_valuations.csv',
            'game_events': 'game_events.csv',
            'game_lineups': 'game_lineups.csv',
            'club_games': 'club_games.csv'
        }
        
        for name, filename in csv_files.items():
            file_path = self.data_dir / filename
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    
                    # Apply data cleaning based on dataset type
                    if name == 'players':
                        df = self.clean_players_data(df)
                    elif name == 'clubs':
                        df = self.clean_clubs_data(df)
                    
                    self.dataframes[name] = df
                    logger.info(f"Loaded {len(df)} records from {filename}")
                except Exception as e:
                    logger.error(f"Failed loading {filename}: {e}")
        
        # Create competition lookup after loading
        self.create_competition_lookup()
        
        logger.info(f"Loaded {len(self.dataframes)} datasets")

    def clean_players_data(self, df):
        """Clean players dataset"""
        # Clean market values
        if 'market_value_in_eur' in df.columns:
            df['market_value_in_eur'] = df['market_value_in_eur'].apply(self.clean_market_value)
        
        if 'highest_market_value_in_eur' in df.columns:
            df['highest_market_value_in_eur'] = df['highest_market_value_in_eur'].apply(self.clean_market_value)
        
        # Clean club names
        if 'current_club_name' in df.columns:
            df['current_club_name'] = df['current_club_name'].apply(self.clean_club_name)
        
        # Clean player names
        for name_col in ['name', 'first_name', 'last_name']:
            if name_col in df.columns:
                df[name_col] = df[name_col].fillna('').astype(str).str.strip()
        
        # Clean other text fields
        text_fields = ['position', 'sub_position', 'foot', 'country_of_citizenship', 
                      'city_of_birth', 'country_of_birth', 'agent_name']
        for field in text_fields:
            if field in df.columns:
                df[field] = df[field].fillna('').astype(str).str.strip()
        
        return df

    def clean_clubs_data(self, df):
        """Clean clubs dataset"""
        # Clean club names
        if 'name' in df.columns:
            df['name'] = df['name'].apply(self.clean_club_name)
        
        # Clean market values
        if 'total_market_value' in df.columns:
            df['total_market_value'] = df['total_market_value'].apply(self.clean_market_value)
        
        # Clean transfer records
        if 'net_transfer_record' in df.columns:
            df['net_transfer_record'] = df['net_transfer_record'].apply(self.clean_market_value)
        
        return df

    def filter_current_players(self):
        """Filter players dataset to only include currently active players (last_season = 2024)"""
        if 'players' not in self.dataframes:
            logger.error("Players dataset not loaded!")
            return
        
        original_count = len(self.dataframes['players'])
        
        # Filter for players with last_season = 2024 (currently active)
        current_players = self.dataframes['players'][
            self.dataframes['players']['last_season'] == 2024
        ].copy()
        
        # Additional filters for data quality
        # Remove players without basic information
        current_players = current_players[
            (current_players['name'].notna()) & 
            (current_players['name'] != '') &
            (current_players['position'].notna()) &
            (current_players['position'] != '')
        ].copy()
        
        # Remove players with unrealistic ages (if date_of_birth is available)
        if 'date_of_birth' in current_players.columns:
            current_players = current_players[
                (current_players['date_of_birth'].isna()) |  # Keep if no DOB data
                (pd.to_datetime(current_players['date_of_birth'], errors='coerce').notna())  # Valid dates only
            ].copy()
        
        # Filter out players with extremely low or no market value (likely inactive)
        if 'market_value_in_eur' in current_players.columns:
            current_players = current_players[
                (current_players['market_value_in_eur'].isna()) |  # Keep if no market value data
                (current_players['market_value_in_eur'] >= 25000)  # Minimum €25k market value
            ].copy()
        
        self.dataframes['players'] = current_players
        
        logger.info(f"Filtered players: {original_count:,} → {len(current_players):,} currently active players")
        logger.info(f"Removed {original_count - len(current_players):,} retired/inactive players")

    def optimize_dataframe(self, df):
        """Optimize dataframe memory usage and clean data"""
        # Convert object columns to category where appropriate
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')
        
        # Handle missing values
        df = df.fillna('')
        
        return df
    
    def calculate_player_career_stats(self, player_id):
        """Calculate comprehensive career statistics for a player"""
        if 'appearances' not in self.dataframes:
            return {}
        
        player_apps = self.dataframes['appearances'][
            self.dataframes['appearances']['player_id'] == player_id
        ]
        
        if player_apps.empty:
            return {}
        
        # Filter recent appearances only (last 5 years for better relevance)
        current_year = datetime.now().year
        if 'date' in player_apps.columns:
            player_apps = player_apps[
                pd.to_datetime(player_apps['date'], errors='coerce').dt.year >= (current_year - 5)
            ]
        
        # Basic career stats
        total_appearances = len(player_apps)
        total_goals = player_apps['goals'].sum()
        total_assists = player_apps['assists'].sum()
        total_minutes = player_apps['minutes_played'].sum()
        total_yellow_cards = player_apps['yellow_cards'].sum()
        total_red_cards = player_apps['red_cards'].sum()
        
        career_stats = {
            'total_appearances': int(total_appearances),
            'total_goals': int(total_goals),
            'total_assists': int(total_assists),
            'total_minutes': int(total_minutes),
            'total_yellow_cards': int(total_yellow_cards),
            'total_red_cards': int(total_red_cards),
            'goals_per_appearance': 0.0,
            'assists_per_appearance': 0.0,
            'minutes_per_appearance': 0.0,
            'goal_contributions_per_90': 0.0,
            'discipline_score': 0.0
        }
        
        # Calculate performance ratios
        if total_appearances > 0:
            career_stats['goals_per_appearance'] = round(total_goals / total_appearances, 4)
            career_stats['assists_per_appearance'] = round(total_assists / total_appearances, 4)
            career_stats['minutes_per_appearance'] = round(total_minutes / total_appearances, 1)
            
            # Discipline score (lower is better)
            career_stats['discipline_score'] = round(
                (total_yellow_cards + (total_red_cards * 2)) / total_appearances, 3
            )
        
        # Goal contributions per 90 minutes
        if total_minutes > 0:
            goal_contributions = total_goals + total_assists
            career_stats['goal_contributions_per_90'] = round(
                (goal_contributions * 90) / total_minutes, 3
            )
        
        # Enhanced competition analysis with improved league mapping
        career_stats['competition_breakdown'] = self.analyze_competition_experience(player_apps)
        career_stats['league_experience'] = self.extract_league_experience(career_stats['competition_breakdown'])
        
        # Recent form analysis (last 10 appearances)
        career_stats['recent_form'] = self.analyze_recent_form(player_apps)
        
        # Performance trends
        career_stats['performance_trends'] = self.analyze_performance_trends(player_apps)
        
        return career_stats
    
    def analyze_competition_experience(self, player_apps):
        """Analyze player's competition experience with improved competition mapping"""
        comp_stats = []
        
        for comp_id, comp_data in player_apps.groupby('competition_id'):
            # Get competition info from lookup
            comp_info = self.competition_lookup.get(comp_id, {})
            comp_name = comp_info.get('name', str(comp_id))
            comp_type = comp_info.get('type', 'unknown')
            comp_country = comp_info.get('country', '')
            is_major = comp_info.get('is_major', False)
            
            comp_stat = {
                'competition_id': str(comp_id),
                'competition_name': str(comp_name),
                'competition_type': comp_type,
                'country': comp_country,
                'is_major_league': is_major,
                'appearances': len(comp_data),
                'goals': int(comp_data['goals'].sum()),
                'assists': int(comp_data['assists'].sum()),
                'minutes': int(comp_data['minutes_played'].sum()),
                'goals_per_game': round(comp_data['goals'].sum() / len(comp_data), 3),
                'assists_per_game': round(comp_data['assists'].sum() / len(comp_data), 3)
            }
            comp_stats.append(comp_stat)
        
        # Sort by importance (major leagues first, then by appearances and goals)
        comp_stats.sort(key=lambda x: (x['is_major_league'], x['appearances'], x['goals']), reverse=True)
        return comp_stats
    
    def extract_league_experience(self, competition_breakdown):
        """Extract league experience for RAG filtering with improved mapping"""
        league_experience = {
            'top_5_leagues': [],
            'major_leagues': [],
            'european_competitions': [],
            'total_leagues': 0
        }
        
        for comp in competition_breakdown:
            comp_id = comp['competition_id']
            comp_name = comp['competition_name'].lower()
            comp_type = comp.get('competition_type', '')
            
            # Check if it's a top 5 league
            top_5_ids = ['GB1', 'ES1', 'L1', 'IT1', 'FR1']
            if comp_id in top_5_ids:
                league_name = {
                    'GB1': 'premier league',
                    'ES1': 'la liga', 
                    'L1': 'bundesliga',
                    'IT1': 'serie a',
                    'FR1': 'ligue 1'
                }.get(comp_id, comp_name)
                
                league_experience['top_5_leagues'].append({
                    'league': league_name,
                    'appearances': comp['appearances'],
                    'goals': comp['goals'],
                    'assists': comp['assists']
                })
            
            # Check for European competitions
            elif comp_type in ['uefa_champions_league', 'europa_league', 'uefa_europa_conference_league'] or comp_id in ['CL', 'EL', 'UCOL']:
                comp_name_clean = {
                    'CL': 'champions league',
                    'EL': 'europa league',
                    'UCOL': 'conference league'
                }.get(comp_id, comp_name)
                
                league_experience['european_competitions'].append({
                    'competition': comp_name_clean,
                    'appearances': comp['appearances'],
                    'goals': comp['goals']
                })
            
            # Other major leagues
            elif comp_type == 'domestic_league' and comp['appearances'] >= 5:
                league_experience['major_leagues'].append({
                    'league': comp_name,
                    'appearances': comp['appearances'],
                    'goals': comp['goals']
                })
        
        league_experience['total_leagues'] = len(set([
            item['league'] for item in league_experience['top_5_leagues'] + league_experience['major_leagues']
        ]))
        
        return league_experience
    
    def analyze_recent_form(self, player_apps):
        """Analyze recent form with date sorting"""
        # Sort by date and get last 10 appearances
        if 'date' in player_apps.columns:
            recent_apps = player_apps.sort_values('date').tail(10)
        else:
            recent_apps = player_apps.tail(10)
        
        if len(recent_apps) == 0:
            return {}
        
        return {
            'appearances': len(recent_apps),
            'goals': int(recent_apps['goals'].sum()),
            'assists': int(recent_apps['assists'].sum()),
            'minutes': int(recent_apps['minutes_played'].sum()),
            'goals_per_game': round(recent_apps['goals'].sum() / len(recent_apps), 3),
            'assists_per_game': round(recent_apps['assists'].sum() / len(recent_apps), 3),
            'avg_minutes': round(recent_apps['minutes_played'].mean(), 1)
        }
    
    def analyze_performance_trends(self, player_apps):
        """Analyze performance trends over time"""
        if len(player_apps) < 10:
            return {'trend': 'insufficient_data'}
        
        # Sort by date if available
        if 'date' in player_apps.columns:
            apps_sorted = player_apps.sort_values('date')
        else:
            apps_sorted = player_apps
        
        # Split into first half and second half of career
        mid_point = len(apps_sorted) // 2
        first_half = apps_sorted.iloc[:mid_point]
        second_half = apps_sorted.iloc[mid_point:]
        
        first_goals_per_game = first_half['goals'].sum() / len(first_half) if len(first_half) > 0 else 0
        second_goals_per_game = second_half['goals'].sum() / len(second_half) if len(second_half) > 0 else 0
        
        # Determine trend
        if second_goals_per_game > first_goals_per_game * 1.2:
            trend = 'improving'
        elif second_goals_per_game < first_goals_per_game * 0.8:
            trend = 'declining'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'early_career_goals_per_game': round(first_goals_per_game, 3),
            'recent_career_goals_per_game': round(second_goals_per_game, 3),
            'improvement_factor': round(second_goals_per_game / max(first_goals_per_game, 0.001), 2)
        }
    
    def calculate_transfer_history(self, player_id):
        """Calculate transfer history and patterns"""
        if 'transfers' not in self.dataframes:
            return {}
        
        player_transfers = self.dataframes['transfers'][
            self.dataframes['transfers']['player_id'] == player_id
        ].sort_values('transfer_date')
        
        if player_transfers.empty:
            return {}
        
        # Filter recent transfers only (last 10 years)
        current_year = datetime.now().year
        player_transfers = player_transfers[
            pd.to_datetime(player_transfers['transfer_date'], errors='coerce').dt.year >= (current_year - 10)
        ]
        
        # Clean transfer fees (handle NaN and invalid values)
        transfer_fees = player_transfers['transfer_fee'].apply(self.clean_market_value)
        market_values = player_transfers['market_value_in_eur'].apply(self.clean_market_value)
        
        transfer_info = {
            'total_transfers': len(player_transfers),
            'total_transfer_fees': float(transfer_fees.sum()),
            'average_transfer_fee': float(transfer_fees.mean()) if len(transfer_fees) > 0 else 0,
            'highest_transfer_fee': float(transfer_fees.max()) if len(transfer_fees) > 0 else 0,
            'transfer_frequency': 0.0,
            'career_trajectory': 'unknown',
            'recent_transfers': []
        }
        
        # Calculate transfer frequency
        if len(player_transfers) > 1:
            try:
                first_transfer = pd.to_datetime(player_transfers['transfer_date'].iloc[0])
                last_transfer = pd.to_datetime(player_transfers['transfer_date'].iloc[-1])
                years_span = (last_transfer - first_transfer).days / 365.25
                transfer_info['transfer_frequency'] = round(years_span / (len(player_transfers) - 1), 2) if years_span > 0 else 0
            except:
                transfer_info['transfer_frequency'] = 0
        
        # Career trajectory analysis
        valid_market_values = market_values[market_values > 0]
        if len(valid_market_values) > 1:
            if valid_market_values.iloc[-1] > valid_market_values.iloc[0] * 1.3:
                transfer_info['career_trajectory'] = 'upward'
            elif valid_market_values.iloc[-1] < valid_market_values.iloc[0] * 0.7:
                transfer_info['career_trajectory'] = 'downward'
            else:
                transfer_info['career_trajectory'] = 'stable'
        
        # Recent transfers (last 3)
        recent = player_transfers.tail(3)
        for _, transfer in recent.iterrows():
            transfer_info['recent_transfers'].append({
                'date': str(transfer['transfer_date']),
                'from_club': self.clean_club_name(transfer.get('from_club_name', '')),
                'to_club': self.clean_club_name(transfer.get('to_club_name', '')),
                'fee': float(transfer.get('transfer_fee', 0)) if pd.notna(transfer.get('transfer_fee')) else 0,
                'market_value': float(transfer.get('market_value_in_eur', 0)) if pd.notna(transfer.get('market_value_in_eur')) else 0
            })
        
        return transfer_info
    
    def calculate_market_value_trends(self, player_id):
        """Calculate market value trends and patterns"""
        if 'player_valuations' not in self.dataframes:
            return {}
        
        valuations = self.dataframes['player_valuations'][
            self.dataframes['player_valuations']['player_id'] == player_id
        ].sort_values('date')
        
        if valuations.empty:
            return {}
        
        # Filter recent valuations only (last 5 years)
        current_year = datetime.now().year
        valuations = valuations[
            pd.to_datetime(valuations['date'], errors='coerce').dt.year >= (current_year - 5)
        ]
        
        # Clean market value data
        clean_values = valuations['market_value_in_eur'].apply(self.clean_market_value)
        clean_values = clean_values[clean_values > 0]  # Remove zero/negative values
        
        if len(clean_values) == 0:
            return {}
        
        market_info = {
            'current_market_value': float(clean_values.iloc[-1]),
            'peak_market_value': float(clean_values.max()),
            'lowest_market_value': float(clean_values.min()),
            'value_trend': 'stable',
            'value_volatility': 0.0,
            'recent_value_change': 0.0,
            'value_growth_rate': 0.0
        }
        
        if len(clean_values) > 1:
            # Calculate trend (last 3 values vs first 3 values)
            recent_values = clean_values.tail(min(3, len(clean_values)))
            early_values = clean_values.head(min(3, len(clean_values)))
            
            if len(recent_values) > 0 and len(early_values) > 0:
                recent_avg = recent_values.mean()
                early_avg = early_values.mean()
                trend_change = (recent_avg - early_avg) / early_avg
                
                if trend_change > 0.15:
                    market_info['value_trend'] = 'rising'
                elif trend_change < -0.15:
                    market_info['value_trend'] = 'declining'
                else:
                    market_info['value_trend'] = 'stable'
                
                market_info['value_growth_rate'] = round(trend_change, 3)
            
            # Calculate volatility (coefficient of variation)
            market_info['value_volatility'] = round(clean_values.std() / clean_values.mean(), 3)
            
            # Recent change (last two values)
            if len(clean_values) >= 2:
                market_info['recent_value_change'] = round(
                    (clean_values.iloc[-1] - clean_values.iloc[-2]) / clean_values.iloc[-2], 3
                )
        
        return market_info
    
    def get_playing_style_indicators(self, player_id):
        """Derive playing style indicators from performance data"""
        career_stats = self.calculate_player_career_stats(player_id)
        
        if not career_stats:
            return {}
        
        style_indicators = {
            'attacking_threat': 0.0,
            'creativity': 0.0,
            'consistency': 0.0,
            'experience_level': 'unknown',
            'goal_scoring_ability': 'unknown',
            'discipline': 'unknown',
            'versatility': 0,
            'big_game_experience': False,
            'international_experience': False
        }
        
        # Attacking threat (goal contributions per 90)
        style_indicators['attacking_threat'] = career_stats.get('goal_contributions_per_90', 0)
        
        # Creativity (assist ratio)
        total_goals = career_stats.get('total_goals', 0)
        total_assists = career_stats.get('total_assists', 0)
        if total_goals + total_assists > 0:
            style_indicators['creativity'] = round(total_assists / (total_goals + total_assists), 3)
        
        # Consistency (minutes per appearance ratio)
        minutes_per_app = career_stats.get('minutes_per_appearance', 0)
        style_indicators['consistency'] = round(min(minutes_per_app / 90, 1.0), 3)
        
        # Experience level (adjusted for current players)
        appearances = career_stats.get('total_appearances', 0)
        if appearances > 250:
            style_indicators['experience_level'] = 'veteran'
        elif appearances > 100:
            style_indicators['experience_level'] = 'experienced'
        elif appearances > 50:
            style_indicators['experience_level'] = 'developing'
        elif appearances > 15:
            style_indicators['experience_level'] = 'emerging'
        else:
            style_indicators['experience_level'] = 'young'
        
        # Goal scoring ability
        goals_per_game = career_stats.get('goals_per_appearance', 0)
        if goals_per_game > 0.8:
            style_indicators['goal_scoring_ability'] = 'world_class'
        elif goals_per_game > 0.6:
            style_indicators['goal_scoring_ability'] = 'prolific'
        elif goals_per_game > 0.4:
            style_indicators['goal_scoring_ability'] = 'regular'
        elif goals_per_game > 0.2:
            style_indicators['goal_scoring_ability'] = 'occasional'
        elif goals_per_game > 0.05:
            style_indicators['goal_scoring_ability'] = 'rare'
        else:
            style_indicators['goal_scoring_ability'] = 'non_scorer'
        
        # Discipline
        discipline_score = career_stats.get('discipline_score', 0)
        if discipline_score > 0.4:
            style_indicators['discipline'] = 'poor'
        elif discipline_score > 0.25:
            style_indicators['discipline'] = 'questionable'
        elif discipline_score > 0.15:
            style_indicators['discipline'] = 'average'
        elif discipline_score > 0.05:
            style_indicators['discipline'] = 'good'
        else:
            style_indicators['discipline'] = 'excellent'
        
        # Versatility (number of competitions)
        comp_breakdown = career_stats.get('competition_breakdown', [])
        style_indicators['versatility'] = len(comp_breakdown)
        
        # Big game experience
        league_exp = career_stats.get('league_experience', {})
        european_comps = league_exp.get('european_competitions', [])
        top_5_leagues = league_exp.get('top_5_leagues', [])
        
        style_indicators['big_game_experience'] = len(european_comps) > 0 or len(top_5_leagues) > 0
        style_indicators['international_experience'] = len(top_5_leagues) > 0
        
        return style_indicators
    
    def create_comprehensive_player_profile(self, player_id):
        """Create comprehensive player profile combining all data sources"""
        if 'players' not in self.dataframes:
            return {}
        
        player_info = self.dataframes['players'][
            self.dataframes['players']['player_id'] == player_id
        ]
        
        if player_info.empty:
            return {}
        
        player_row = player_info.iloc[0]
        
        profile = {
            # Basic Info
            'player_id': int(player_id),
            'name': str(player_row.get('name', '')).strip(),
            'first_name': str(player_row.get('first_name', '')).strip(),
            'last_name': str(player_row.get('last_name', '')).strip(),
            'date_of_birth': str(player_row.get('date_of_birth', '')),
            'age': self.calculate_age(player_row.get('date_of_birth')),
            'nationality': str(player_row.get('country_of_citizenship', '')).strip(),
            'birth_place': self.clean_birth_place(player_row),
            
            # Physical & Playing Info
            'position': str(player_row.get('position', '')).strip(),
            'sub_position': str(player_row.get('sub_position', '')).strip(),
            'preferred_foot': str(player_row.get('foot', '')).strip(),
            'height_cm': float(player_row.get('height_in_cm', 0)) if pd.notna(player_row.get('height_in_cm')) else 0,
            
            # Current Club Info
            'current_club_id': player_row.get('current_club_id'),
            'current_club_name': self.clean_club_name(player_row.get('current_club_name', '')),
            'contract_expiration': str(player_row.get('contract_expiration_date', '')),
            'agent_name': str(player_row.get('agent_name', '')).strip(),
            
            # Market Value - using cleaned values
            'current_market_value': self.clean_market_value(player_row.get('market_value_in_eur', 0)),
            'highest_market_value': self.clean_market_value(player_row.get('highest_market_value_in_eur', 0)),
            
            # Enhanced Analytics
            'career_stats': self.calculate_player_career_stats(player_id),
            'transfer_history': self.calculate_transfer_history(player_id),
            'market_value_trends': self.calculate_market_value_trends(player_id),
            'playing_style': self.get_playing_style_indicators(player_id),
        }
        
        # Add enhanced club context
        profile['club_context'] = self.get_club_context(profile['current_club_id'])
        
        return profile
    
    def clean_birth_place(self, player_row):
        """Clean birth place data"""
        city = str(player_row.get('city_of_birth', '')).strip()
        country = str(player_row.get('country_of_birth', '')).strip()
        
        if city and country and city.lower() != 'nan' and country.lower() != 'nan':
            return f"{city}, {country}"
        elif country and country.lower() != 'nan':
            return country
        elif city and city.lower() != 'nan':
            return city
        else:
            return ""
    
    def calculate_age(self, birth_date):
        """Calculate age from birth date"""
        if pd.isna(birth_date) or birth_date == '' or str(birth_date).lower() == 'nan':
            return 0
        
        try:
            birth = pd.to_datetime(birth_date)
            today = pd.Timestamp.now()
            age = int((today - birth).days / 365.25)
            
            # Validate age for current players (reasonable range)
            if 16 <= age <= 45:
                return age
            else:
                return 0
        except:
            return 0
        
    def get_club_context(self, club_id):
        """Get enhanced club context"""
        if not club_id or 'clubs' not in self.dataframes:
            return {}
        
        club_info = self.dataframes['clubs'][
            self.dataframes['clubs']['club_id'] == club_id
        ]
        
        if club_info.empty:
            return {}
        
        club_row = club_info.iloc[0]
        return {
            'club_name': self.clean_club_name(club_row.get('name', '')),
            'league': str(club_row.get('domestic_competition_id', '')),
            'club_market_value': self.clean_market_value(club_row.get('total_market_value', 0)),
            'squad_size': int(club_row.get('squad_size', 0)) if pd.notna(club_row.get('squad_size')) else 0,
            'stadium': str(club_row.get('stadium_name', '')).strip(),
            'coach': str(club_row.get('coach_name', '')).strip()
        }
    
    def create_scouting_text(self, profile):
        """Create rich text description for embedding"""
        text_parts = []
        
        # Basic identity
        name = profile.get('name', 'Unknown Player')
        age = profile.get('age', 0)
        nationality = profile.get('nationality', 'Unknown')
        position = profile.get('position', 'Unknown')
        sub_position = profile.get('sub_position', '')
        position_desc = f"{position}"
        if sub_position and sub_position != position and sub_position.lower() != 'nan':
            position_desc += f" ({sub_position})"
        
        text_parts.append(f"{name} is a {age}-year-old {position_desc} from {nationality}")
        
        # Current situation
        club = profile.get('current_club_name', '')
        if club and club.lower() != 'nan':
            text_parts.append(f"currently playing for {club}")
        
        # Physical attributes
        height = profile.get('height_cm', 0)
        foot = profile.get('preferred_foot', '')
        physical_desc = []
        if height > 0:
            if height > 190:
                physical_desc.append(f"tall {height}cm")
            elif height < 170:
                physical_desc.append(f"compact {height}cm")
            else:
                physical_desc.append(f"{height}cm")
        if foot and foot.lower() not in ['nan', '']:
            physical_desc.append(f"{foot}-footed")
        if physical_desc:
            text_parts.append(f"Physical: {', '.join(physical_desc)}")
        
        # Career statistics
        career_stats = profile.get('career_stats', {})
        if career_stats:
            appearances = career_stats.get('total_appearances', 0)
            goals = career_stats.get('total_goals', 0)
            assists = career_stats.get('total_assists', 0)
            
            if appearances > 0:
                text_parts.append(f"Career: {appearances} apps, {goals} goals, {assists} assists")
                
                # Performance ratios with context
                goals_per_game = career_stats.get('goals_per_appearance', 0)
                assists_per_game = career_stats.get('assists_per_appearance', 0)
                goal_contrib_per_90 = career_stats.get('goal_contributions_per_90', 0)
                
                performance_desc = f"{goals_per_game:.2f} goals/game, {assists_per_game:.2f} assists/game"
                if goal_contrib_per_90 > 0:
                    performance_desc += f", {goal_contrib_per_90:.2f} contributions/90min"
                text_parts.append(f"Performance: {performance_desc}")
        
        # Playing style
        playing_style = profile.get('playing_style', {})
        if playing_style:
            experience = playing_style.get('experience_level', '')
            goal_ability = playing_style.get('goal_scoring_ability', '')
            discipline = playing_style.get('discipline', '')
            big_game_exp = playing_style.get('big_game_experience', False)
            
            style_desc = []
            if experience and experience != 'unknown':
                style_desc.append(f"{experience} player")
            if goal_ability and goal_ability != 'unknown':
                style_desc.append(f"{goal_ability} goalscorer")
            if discipline and discipline != 'unknown':
                style_desc.append(f"{discipline} discipline")
            if big_game_exp:
                style_desc.append("big game experience")
            
            if style_desc:
                text_parts.append(f"Profile: {', '.join(style_desc)}")
        
        # Enhanced league experience with proper league names
        league_exp = career_stats.get('league_experience', {})
        if league_exp:
            top_5_leagues = league_exp.get('top_5_leagues', [])
            european_comps = league_exp.get('european_competitions', [])
            
            experience_desc = []
            if top_5_leagues:
                leagues = [league['league'] for league in top_5_leagues]
                experience_desc.append(f"Top 5 leagues: {', '.join(leagues)}")
            
            if european_comps:
                comps = [comp['competition'] for comp in european_comps]
                experience_desc.append(f"European: {', '.join(comps)}")
            
            if experience_desc:
                text_parts.append(f"Experience: {'; '.join(experience_desc)}")
        
        # Market value with context
        current_value = profile.get('current_market_value', 0)
        peak_value = profile.get('highest_market_value', 0)
        
        if current_value > 0:
            if current_value >= 100_000_000:
                text_parts.append(f"Elite market value: €{current_value:,}")
            elif current_value >= 50_000_000:
                text_parts.append(f"High market value: €{current_value:,}")
            elif current_value >= 10_000_000:
                text_parts.append(f"Significant market value: €{current_value:,}")
            else:
                text_parts.append(f"Market value: €{current_value:,}")
        
        if peak_value > current_value * 1.5 and peak_value > 0:
            text_parts.append(f"Peak value: €{peak_value:,}")
        
        # Market trends with context
        market_trends = profile.get('market_value_trends', {})
        if market_trends:
            trend = market_trends.get('value_trend', '')
            if trend == 'rising':
                text_parts.append("Value trending upward")
            elif trend == 'declining':
                text_parts.append("Value declining")
        
        # Contract and availability
        contract_end = profile.get('contract_expiration', '')
        if contract_end and contract_end != '' and 'nan' not in str(contract_end).lower():
            try:
                contract_date = pd.to_datetime(contract_end)
                current_date = pd.Timestamp.now()
                months_remaining = (contract_date - current_date).days / 30
                
                if months_remaining < 6:
                    text_parts.append("Contract expiring soon")
                elif months_remaining < 18:
                    text_parts.append("Contract expires within 18 months")
            except:
                pass
        
        # Agent representation
        agent = profile.get('agent_name', '')
        if agent and agent != '' and 'nan' not in str(agent).lower():
            text_parts.append(f"Agent: {agent}")
        
        return '. '.join(text_parts) + '.'
    
    def process_all_players(self, limit):
        """Process all players to create comprehensive profiles"""
        
        if 'players' not in self.dataframes:
            logger.error("Players dataset not loaded!")
            return []
        
        # Filter for current players first
        self.filter_current_players()
        
        players_df = self.dataframes['players']
        if limit:
            players_df = players_df.head(limit)
        
        logger.info(f"Processing {len(players_df):,} current players...")
        
        processed_players = []
        error_count = 0
        
        for idx, (_, player_row) in enumerate(players_df.iterrows()):
            if idx % 1000 == 0:
                logger.info(f"Progress: {idx:,}/{len(players_df):,} players processed ({len(processed_players):,} successful)")
            
            try:
                player_id = player_row['player_id']
                profile = self.create_comprehensive_player_profile(player_id)
                
                if profile and profile.get('name'):
                    # Additional quality check - ensure player has some career activity
                    career_stats = profile.get('career_stats', {})
                    appearances = career_stats.get('total_appearances', 0)
                    
                    # Only include players with some career activity or high market value
                    if appearances >= 5 or profile.get('current_market_value', 0) >= 100000:
                        # Create enhanced embedding text
                        embedding_text = self.create_scouting_text(profile)
                        
                        # Create final player record optimized for RAG
                        player_record = {
                            'player_id': int(player_id),
                            'name': profile.get('name', ''),
                            'position': profile.get('position', ''),
                            'age': profile.get('age', 0),
                            'nationality': profile.get('nationality', ''),
                            'current_club': profile.get('current_club_name', ''),
                            'market_value': profile.get('current_market_value', 0),
                            'embedding_text': embedding_text,
                            'metadata': profile
                        }
                        
                        processed_players.append(player_record)
                
            except Exception as e:
                error_count += 1
                if error_count <= 10:
                    logger.warning(f"Error processing player {player_row.get('player_id', 'unknown')}: {e}")
                continue
        
        if error_count > 10:
            logger.warning(f"Total errors: {error_count} (showing first 10 only)")
        
        logger.info(f"Successfully processed {len(processed_players):,} current players")
        return processed_players

class SoccerEmbeddingGenerator:
    """Generate embeddings for soccer player profiles"""
    
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        
    def initialize_model(self):
        """Initialize the embedding model"""
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        
    def generate_embeddings(self, player_records):
        """Generate embeddings for player records"""
        
        if self.model is None:
            self.initialize_model()
        
        logger.info(f"Generating embeddings for {len(player_records):,} players...")
        
        # Extract texts for batch processing
        texts = []
        valid_indices = []
        
        for i, record in enumerate(player_records):
            text = record.get('embedding_text', '').strip()
            if text and len(text) > 10:
                texts.append(text)
                valid_indices.append(i)
            else:
                logger.warning(f"Skipping player {record.get('name', 'unknown')} - insufficient text")
        
        logger.info(f"Processing {len(texts):,} valid texts")
        
        if not texts:
            logger.error("No valid texts to process")
            return player_records
        
        # Generate embeddings in batches
        batch_size = 32
        all_embeddings = []
        
        try:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(texts) + batch_size - 1) // batch_size
                
                logger.info(f"Processing batch {batch_num}/{total_batches}")
                
                # Generate embeddings with error handling
                try:
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        show_progress_bar=False,
                        batch_size=min(16, len(batch_texts)),
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    )
                    all_embeddings.extend(batch_embeddings.tolist())
                    
                except Exception as e:
                    logger.error(f"Batch {batch_num} failed: {e}")
                    # Add zero embeddings for failed batch
                    embedding_dim = 384
                    for _ in batch_texts:
                        all_embeddings.append([0.0] * embedding_dim)
            
            # Assign embeddings back to valid records
            embedding_idx = 0
            for record_idx in valid_indices:
                if embedding_idx < len(all_embeddings):
                    player_records[record_idx]['embedding'] = all_embeddings[embedding_idx]
                    embedding_idx += 1
                else:
                    player_records[record_idx]['embedding'] = [0.0] * 384
            
            logger.info(f"Generated embeddings for {len(all_embeddings):,} players")
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
        
        return player_records
    
    def save_embeddings(self, player_records, output_dir):
        """Save embeddings and metadata"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Filter records with valid embeddings
        valid_records = [
            record for record in player_records 
            if 'embedding' in record and record['embedding'] and len(record['embedding']) > 0
        ]
        
        if not valid_records:
            logger.error("No valid embeddings to save")
            return {}
        
        logger.info(f"Saving {len(valid_records):,} valid player records")
        
        # Save embeddings as numpy arrays for fast loading
        embeddings = np.array([record['embedding'] for record in valid_records])
        np.save(output_path / "player_embeddings.npy", embeddings)
        
        # Save metadata (only for valid records)
        metadata = []
        for record in valid_records:
            metadata.append({
                'player_id': record['player_id'],
                'name': record['name'],
                'position': record['position'],
                'age': record['age'],
                'nationality': record['nationality'],
                'current_club': record['current_club'],
                'market_value': record['market_value'],
                'embedding_text': record['embedding_text']
            })
        
        # Save as JSON
        with open(output_path / "player_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Save as CSV for easy viewing
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(output_path / "player_metadata.csv", index=False)
        
        # Save detailed profiles
        detailed_profiles = [record['metadata'] for record in valid_records]
        with open(output_path / "detailed_player_profiles.json", 'w', encoding='utf-8') as f:
            json.dump(detailed_profiles, f, indent=2, ensure_ascii=False, default=str)
        
        # Create enhanced summary statistics
        summary_stats = {
            'total_players': len(valid_records),
            'embedding_dimension': len(valid_records[0]['embedding']) if valid_records else 0,
            'data_quality': {
                'avg_text_length': np.mean([len(r['embedding_text']) for r in valid_records]),
                'players_with_stats': len([r for r in valid_records if r['metadata'].get('career_stats', {}).get('total_appearances', 0) > 0]),
                'players_with_market_value': len([r for r in valid_records if r['market_value'] > 0]),
                'active_players_only': True,
                'last_season_filter': 2024
            },
            'position_distribution': metadata_df['position'].value_counts().to_dict(),
            'top_clubs': metadata_df['current_club'].value_counts().head(20).to_dict(),
            'age_distribution': {
                'mean': round(metadata_df['age'].mean(), 1),
                'median': int(metadata_df['age'].median()),
                'min': int(metadata_df['age'].min()),
                'max': int(metadata_df['age'].max()),
                'std': round(metadata_df['age'].std(), 1)
            },
            'market_value_stats': {
                'mean': round(metadata_df['market_value'].mean(), 0),
                'median': round(metadata_df['market_value'].median(), 0),
                'max': int(metadata_df['market_value'].max()),
                'players_over_1m': len(metadata_df[metadata_df['market_value'] > 1_000_000]),
                'players_over_10m': len(metadata_df[metadata_df['market_value'] > 10_000_000]),
                'players_over_50m': len(metadata_df[metadata_df['market_value'] > 50_000_000])
            },
            'nationality_distribution': metadata_df['nationality'].value_counts().head(20).to_dict(),
            'created_at': datetime.now().isoformat(),
            'model_used': self.model_name
        }
        
        # Save summary
        with open(output_path / "embedding_summary.json", 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)
        
        logger.info(f"Summary: {summary_stats['total_players']:,} current players, {summary_stats['embedding_dimension']} dimensions")
        logger.info(f"Data quality: {summary_stats['data_quality']['players_with_stats']:,} players with stats")
        logger.info(f"Market values: {summary_stats['market_value_stats']['players_over_1m']:,} players over €1M")
        
        return summary_stats

def create_soccer_embeddings(data_dir, output_dir, model_name, limit_players):
    """Complete pipeline to create soccer player embeddings for current players only"""
    
    logger.info("Starting Embedding Generation for Current Players (2024)")
    
    # Step 1: Process player data
    logger.info("Step 1: Processing current player data...")
    processor = SoccerDataProcessor(data_dir)
    processor.load_datasets()
    
    # Step 2: Create comprehensive profiles
    logger.info("Step 2: Creating comprehensive player profiles...")
    player_records = processor.process_all_players(limit=limit_players)
    
    if not player_records:
        logger.error("No current players found! Check if your dataset has players with last_season = 2024")
        return []
    
    # Step 3: Generate embeddings
    logger.info("Step 3: Generating embeddings...")
    generator = SoccerEmbeddingGenerator(model_name)
    player_records_with_embeddings = generator.generate_embeddings(player_records)
    
    # Step 4: Save results
    logger.info("Step 4: Saving embeddings...")
    summary = generator.save_embeddings(player_records_with_embeddings, output_dir)
    
    logger.info("Complete!")
    logger.info(f"Summary: {summary['total_players']:,} current players, {summary['embedding_dimension']} dimensions")
    logger.info(f"Age range: {summary['age_distribution']['min']}-{summary['age_distribution']['max']} years")
    logger.info(f"Players with market value > €1M: {summary['market_value_stats']['players_over_1m']:,}")
    
    return player_records_with_embeddings

if __name__ == "__main__":
    create_soccer_embeddings(
        data_dir="../data",
        output_dir="../embeddings",
        model_name="all-MiniLM-L6-v2",
        limit_players=None
    )