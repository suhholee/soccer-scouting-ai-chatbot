import os
import streamlit as st
import numpy as np
import json
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
from datetime import datetime
import re
import torch
import warnings
warnings.filterwarnings('ignore') 
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder
import google.generativeai as genai

# Configure logging for Streamlit (optional, Streamlit handles some logging)
logger = logging.getLogger(__name__)

# Set single threading for PyTorch to prevent potential issues in some environments
torch.set_num_threads(1)

EMBEDDINGS_DIR = Path("../embeddings") #adjust if needed
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def initialize_league_mappings() -> Dict[str, List[str]]:
    """Initialize comprehensive league mappings based on actual competition data."""
    return {
        'top 5 leagues': ['GB1', 'ES1', 'L1', 'IT1', 'FR1'],
        'big 5 leagues': ['GB1', 'ES1', 'L1', 'IT1', 'FR1'],
        'top leagues': ['GB1', 'ES1', 'L1', 'IT1', 'FR1'],
        'major leagues': ['GB1', 'ES1', 'L1', 'IT1', 'FR1'],
        'big five': ['GB1', 'ES1', 'L1', 'IT1', 'FR1'],
        'european top leagues': ['GB1', 'ES1', 'L1', 'IT1', 'FR1'],
        
        'premier league': ['GB1', 'premier-league'],
        'epl': ['GB1'], 'english premier league': ['GB1'], 'premier': ['GB1'],
        'england': ['GB1'], 'english league': ['GB1'],
        
        'la liga': ['ES1', 'laliga'],
        'spanish league': ['ES1'], 'spain': ['ES1'], 'laliga': ['ES1'], 'spanish primera': ['ES1'],
        
        'bundesliga': ['L1', 'bundesliga'],
        'german league': ['L1'], 'germany': ['L1'], 'german bundesliga': ['L1'],
        
        'serie a': ['IT1', 'serie-a'],
        'italian league': ['IT1'], 'italy': ['IT1'], 'italian serie a': ['IT1'],
        
        'ligue 1': ['FR1', 'ligue-1'],
        'french league': ['FR1'], 'france': ['FR1'], 'ligue un': ['FR1'],
        
        'eredivisie': ['NL1', 'eredivisie'],
        'dutch league': ['NL1'], 'netherlands': ['NL1'],
        
        'primeira liga': ['PO1', 'liga-portugal-bwin'],
        'portuguese league': ['PO1'], 'portugal': ['PO1'],
        
        'super lig': ['TR1', 'super-lig'],
        'turkish league': ['TR1'], 'turkey': ['TR1'],
        
        'scottish premiership': ['SC1', 'scottish-premiership'],
        'scottish league': ['SC1'], 'scotland': ['SC1'],
        
        'belgian pro league': ['BE1', 'jupiler-pro-league'],
        'belgian league': ['BE1'], 'belgium': ['BE1'],
        
        'super league greece': ['GR1', 'super-league-1'],
        'greek league': ['GR1'], 'greece': ['GR1'],
        
        'superligaen': ['DK1', 'superligaen'],
        'danish league': ['DK1'], 'denmark': ['DK1'],
        
        'premier liga ukraine': ['UKR1', 'premier-liga'],
        'ukrainian league': ['UKR1'], 'ukraine': ['UKR1'],
        
        'russian premier league': ['RU1', 'premier-liga'],
        'russian league': ['RU1'], 'russia': ['RU1'],
        
        'champions league': ['CL', 'uefa-champions-league'],
        'ucl': ['CL'],
        'europa league': ['EL', 'uefa-europa-league'],
        'uel': ['EL'],
        'conference league': ['UCOL', 'uefa-conference-league'],
        'european cups': ['CL', 'EL', 'UCOL']
    }

def clean_market_value_string(value_str: str) -> int:
    """Clean market value strings from queries to integer euros."""
    if not value_str:
        return 0
    try:
        value_str = re.sub(r'[€$£¥]', '', str(value_str))
        value_str = re.sub(r'[+\-±]', '', value_str)
        value_str = value_str.replace(',', '').strip()
        
        if 'm' in value_str.lower() or 'million' in value_str.lower():
            value_str = re.sub(r'(million|mil|m)', '', value_str.lower())
            multiplier = 1_000_000
        elif 'k' in value_str.lower() or 'thousand' in value_str.lower():
            value_str = re.sub(r'(thousand|k)', '', value_str.lower())
            multiplier = 1_000
        else:
            multiplier = 1
        
        numeric_match = re.search(r'(\d+(?:\.\d+)?)', value_str)
        if numeric_match:
            numeric_value = float(numeric_match.group(1))
            return int(numeric_value * multiplier)
        
        return 0
    except (ValueError, TypeError):
        return 0

club_league_mapping = {
    'manchester': 'gb1', 'liverpool': 'gb1', 'arsenal': 'gb1', 'chelsea': 'gb1',
    'tottenham': 'gb1', 'manchester city': 'gb1', 'manchester united': 'gb1',
    'newcastle': 'gb1', 'brighton': 'gb1', 'aston villa': 'gb1', 'west ham': 'gb1',
    'everton': 'gb1', 'crystal palace': 'gb1', 'fulham': 'gb1', 'brentford': 'gb1',
    'wolverhampton': 'gb1', 'nottingham': 'gb1', 'bournemouth': 'gb1',
    'sunderland': 'gb1', 'burnley': 'gb1', 'leeds': 'gb1',
    'real madrid': 'es1', 'barcelona': 'es1', 'atletico': 'es1', 'sevilla': 'es1',
    'villarreal': 'es1', 'real sociedad': 'es1', 'athletic': 'es1', 'valencia': 'es1',
    'betis': 'es1', 'osasuna': 'es1', 'getafe': 'es1', 'alaves': 'es1',
    'rayo vallecano': 'es1', 'mallorca': 'es1', 'las palmas': 'es1', 'cadiz': 'es1',
    'celta vigo': 'es1', 'espanyol': 'es1', 'leganes': 'es1', 'valladolid': 'es1',
    'bayern': 'l1', 'dortmund': 'l1', 'leipzig': 'l1', 'bayer leverkusen': 'l1',
    'eintracht frankfurt': 'l1', 'wolfsburg': 'l1', 'borussia': 'l1', 'stuttgart': 'l1',
    'hoffenheim': 'l1', 'mainz': 'l1', 'augsburg': 'l1', 'heidenheim': 'l1',
    'werder bremen': 'l1', 'freiburg': 'l1', 'union berlin': 'l1', 'cologne': 'l1',
    'hertha': 'l1', 'schalke': 'l1', 'hamburg': 'l1', 'hannover': 'l1',
    'juventus': 'it1', 'milan': 'it1', 'inter': 'it1', 'napoli': 'it1',
    'roma': 'it1', 'lazio': 'it1', 'atalanta': 'it1', 'fiorentina': 'it1',
    'bologna': 'it1', 'torino': 'it1', 'udinese': 'it1', 'sampdoria': 'it1',
    'genoa': 'it1', 'cagliari': 'it1', 'lecce': 'it1', 'verona': 'it1',
    'empoli': 'it1', 'monza': 'it1', 'como': 'it1', 'parma': 'it1',
    'paris': 'fr1', 'marseille': 'fr1', 'lyon': 'fr1', 'monaco': 'fr1',
    'nice': 'fr1', 'lille': 'fr1', 'rennes': 'fr1', 'strasbourg': 'fr1',
    'montpellier': 'fr1', 'nantes': 'fr1', 'toulouse': 'fr1', 'reims': 'fr1',
    'lens': 'fr1', 'brest': 'fr1', 'angers': 'fr1', 'lorient': 'fr1'
}

def parse_query(query: str, league_mappings: Dict[str, List[str]]) -> Dict[str, Any]:
    """Enhanced query parser with comprehensive filter support."""
    query_lower = query.lower().strip()
    filters = {}
    
    logger.info(f"Parsing query: '{query}'")
    
    position_mappings = {
        'centre-forward': ['centre-forward', 'center-forward', 'striker', 'cf', 'st', 'forward'],
        'second striker': ['second striker', 'support striker', 'false 9', 'ss', 'false nine'],
        'left winger': ['left winger', 'left wing', 'lw', 'left-winger'],
        'right winger': ['right winger', 'right wing', 'rw', 'right-winger'],
        'attacking midfield': ['attacking midfielder', 'attacking midfield', 'cam', 'playmaker', 'am', 'number 10', '10'],
        'central midfield': ['central midfielder', 'central midfield', 'midfielder', 'midfield', 'cm', 'box to box'],
        'defensive midfield': ['defensive midfielder', 'defensive midfield', 'cdm', 'holding midfielder', 'dm', '6', 'anchor'],
        'left midfield': ['left midfielder', 'left midfield', 'lm'],
        'right midfield': ['right midfielder', 'right midfield', 'rm'],
        'centre-back': ['centre-back', 'center-back', 'central defender', 'cb', 'centre back', 'center back'],
        'left-back': ['left-back', 'left back', 'lb', 'left back'],
        'right-back': ['right-back', 'right back', 'rb', 'right back'],
        'goalkeeper': ['goalkeeper', 'keeper', 'gk', 'goalie', 'shot stopper'],
        'forward': ['forward', 'striker', 'attacker', 'attack'],
        'winger': ['winger', 'wing', 'wide player', 'wide forward'],
        'midfielder': ['midfielder', 'midfield', 'middle'],
        'defender': ['defender', 'defence', 'defense', 'back'],
    }
    
    position_found = False
    for position_key, position_terms in position_mappings.items():
        if not position_found:
            for term in position_terms:
                if term in query_lower:
                    filters['position'] = position_key
                    logger.info(f"Detected position: {position_key}")
                    position_found = True
                    break
    
    for league_term, league_list in league_mappings.items():
        if league_term in query_lower:
            filters['target_leagues'] = league_list
            logger.info(f"Detected league filter: {league_term} -> {league_list}")
            break
    
    nationality_mappings = {
        'brazilian': 'Brazil', 'argentinian': 'Argentina', 'argentine': 'Argentina',
        'spanish': 'Spain', 'german': 'Germany', 'french': 'France', 'english': 'England',
        'italian': 'Italy', 'portuguese': 'Portugal', 'dutch': 'Netherlands',
        'belgian': 'Belgium', 'croatian': 'Croatia', 'serbian': 'Serbia',
        'polish': 'Poland', 'mexican': 'Mexico', 'colombian': 'Colombia',
        'american': 'United States', 'canadian': 'Canada', 'turkish': 'Turkey', 
        'greek': 'Greece', 'danish': 'Denmark', 'swedish': 'Sweden', 
        'norwegian': 'Norway', 'ukrainian': 'Ukraine', 'russian': 'Russia', 
        'austrian': 'Austria', 'swiss': 'Switzerland', 'uruguayan': 'Uruguay',
        'chilean': 'Chile', 'peruvian': 'Peru', 'japanese': 'Japan',
        'korean': 'South Korea', 'south korean': 'South Korea'
    }
    
    nationality_found = False
    for nat_term, nat_country in nationality_mappings.items():
        if nat_term in query_lower:
            filters['nationality'] = nat_country
            nationality_found = True
            logger.info(f"Detected nationality (adjective): {nat_term} -> {nat_country}")
            break
    
    if not nationality_found:
        country_names = [
            'brazil', 'argentina', 'spain', 'germany', 'france', 'england',
            'italy', 'portugal', 'netherlands', 'belgium', 'croatia', 'serbia',
            'poland', 'mexico', 'colombia', 'united states', 'usa', 'canada', 
            'turkey', 'greece', 'denmark', 'sweden', 'norway', 'ukraine', 
            'russia', 'uruguay', 'chile', 'peru', 'japan', 'south korea'
        ]
        
        for country in country_names:
            if country in query_lower:
                if country == 'usa':
                    filters['nationality'] = 'United States'
                elif country == 'south korea':
                    filters['nationality'] = 'South Korea'
                elif country == 'united states':
                    filters['nationality'] = 'United States'
                else:
                    filters['nationality'] = country.title()
                logger.info(f"Detected nationality (country): {country} -> {filters['nationality']}")
                break
    
    age_patterns = [
        r'under (\d+)', r'younger than (\d+)', r'below (\d+)', r'less than (\d+)',
        r'between (\d+) and (\d+)', r'(\d+) to (\d+) years?', r'(\d+)-(\d+) years?',
        r'(\d+) years? old', r'age (\d+)', r'aged (\d+)', r'over (\d+)', r'above (\d+)',
        r'older than (\d+)', r'more than (\d+) years?', r'(\d+)\+', r'(\d+) plus'
    ]
    
    value_extracted = False
    value_patterns = [
        r'under (\d+(?:\.\d+)?)\s*(?:million|mil|m)(?:\s*euros?|\s*€)?',
        r'below (\d+(?:\.\d+)?)\s*(?:million|mil|m)(?:\s*euros?|\s*€)?',
        r'less than (\d+(?:\.\d+)?)\s*(?:million|mil|m)(?:\s*euros?|\s*€)?',
        r'budget.*?(\d+(?:\.\d+)?)\s*(?:million|mil|m)(?:\s*euros?|\s*€)?',
        r'(\d+(?:\.\d+)?)m?\s*budget',
        r'(\d+(?:\.\d+)?)\s*(?:million|mil|m)(?:\s*euros?|\s*€)?',
        r'€(\d+(?:\.\d+)?)\s*(?:million|mil|m)?',
        r'\$(\d+(?:\.\d+)?)\s*(?:million|mil|m)?',
        r'£(\d+(?:\.\d+)?)\s*(?:million|mil|m)?',
        r'cheap.*?(\d+(?:\.\d+)?)\s*(?:million|mil|m)(?:\s*euros?|\s*€)?',
        r'bargain.*?(\d+(?:\.\d+)?)\s*(?:million|mil|m)(?:\s*euros?|\s*€)?',
        r'affordable.*?(\d+(?:\.\d+)?)\s*(?:million|mil|m)(?:\s*euros?|\s*€)?'
    ]
    
    for pattern in value_patterns:
        match = re.search(pattern, query_lower)
        if match:
            value_str = match.group(1)
            value = clean_market_value_string(f"{value_str}m")
            if any(word in query_lower for word in ['under', 'below', 'less', 'budget', 'cheap', 'bargain', 'affordable']):
                filters['max_market_value'] = value
                logger.info(f"Detected max market value: €{value:,}")
            elif any(word in query_lower for word in ['over', 'above', 'more']):
                filters['min_market_value'] = value
                logger.info(f"Detected min market value: €{value:,}")
            value_extracted = True
            break
    
    if not value_extracted:
        for pattern in age_patterns:
            match = re.search(pattern, query_lower)
            if match:
                if 'between' in pattern or 'to' in pattern or '-' in pattern:
                    ages = [int(x) for x in match.groups() if x]
                    filters['min_age'] = min(ages)
                    filters['max_age'] = max(ages)
                    logger.info(f"Detected age range: {min(ages)}-{max(ages)}")
                else:
                    age = int(match.group(1))
                    if any(word in query_lower for word in ['under', 'below', 'younger', 'less than']):
                        filters['max_age'] = age
                        logger.info(f"Detected max age: {age}")
                    elif any(word in query_lower for word in ['over', 'above', 'older', 'more than', '+', 'plus']):
                        filters['min_age'] = age
                        logger.info(f"Detected min age: {age}")
                    else:
                        filters['target_age'] = age
                        logger.info(f"Detected target age: {age}")
                break
    
    if any(word in query_lower for word in ['young', 'youth']) and 'max_age' not in filters and 'min_age' not in filters:
        filters['max_age'] = 25
        logger.info("Detected 'young' keyword -> max age 25")
    
    performance_keywords = {
        'goalscorer': {'min_goals_per_game': 0.3}, 'prolific': {'min_goals_per_game': 0.4},
        'clinical': {'min_goals_per_game': 0.25}, 'creative': {'min_assists_per_game': 0.2},
        'playmaker': {'min_assists_per_game': 0.25}, 'experienced': {'min_appearances': 100},
        'veteran': {'min_appearances': 200}, 'young talent': {'max_age': 23},
        'prospect': {'max_age': 21}, 'emerging': {'max_age': 23, 'min_appearances': 20},
        'established': {'min_appearances': 150}, 'proven': {'min_appearances': 100},
    }
    
    for keyword, filter_dict in performance_keywords.items():
        if keyword in query_lower:
            filters.update(filter_dict)
            logger.info(f"Detected performance keyword: {keyword}")
    
    if any(word in query_lower for word in ['top', 'best', 'elite', 'world class', 'star', 'superstar']):
        if 'min_market_value' not in filters:
            filters['min_market_value'] = 20_000_000
            logger.info("Detected quality indicator -> min market value €20M")
    
    if any(word in query_lower for word in ['cheap', 'affordable', 'budget', 'low cost', 'bargain', 'value']):
        if 'max_market_value' not in filters:
            filters['max_market_value'] = 15_000_000
            logger.info("Detected budget indicator -> max market value €15M")
    
    if any(word in query_lower for word in ['free agent', 'contract expiring', 'out of contract', 'expiring']):
        filters['contract_status'] = 'expiring'
        logger.info("Detected contract status filter")
    
    logger.info(f"Final extracted filters: {filters}")
    return filters

def create_fallback_strategies(original_filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create smart fallback strategies that prioritize important filters."""
    fallback_strategies = []
    
    if len(original_filters) > 2:
        strategy1 = {}
        for key in ['position', 'min_age', 'max_age', 'target_age', 'min_market_value', 'max_market_value']:
            if key in original_filters:
                strategy1[key] = original_filters[key]
        if strategy1:
            fallback_strategies.append(strategy1)
    
    if 'position' in original_filters:
        strategy2 = {'position': original_filters['position']}
        for key in ['min_market_value', 'max_market_value']:
            if key in original_filters:
                strategy2[key] = original_filters[key]
        fallback_strategies.append(strategy2)
    
    if 'position' in original_filters and 'nationality' in original_filters:
        strategy3 = {
            'position': original_filters['position'],
            'nationality': original_filters['nationality']
        }
        fallback_strategies.append(strategy3)
    
    if 'position' in original_filters:
        strategy4 = {'position': original_filters['position']}
        fallback_strategies.append(strategy4)
    
    strategy5 = {}
    for key in ['nationality', 'min_age', 'max_age', 'target_age', 'min_market_value', 'max_market_value', 'target_leagues']:
        if key in original_filters:
            strategy5[key] = original_filters[key]
    if strategy5:
        fallback_strategies.append(strategy5)
    
    return fallback_strategies

def get_filtered_dataset_indices(filters: Dict[str, Any], player_metadata: List[Dict], detailed_profiles: List[Dict], league_mappings: Dict[str, List[str]]) -> List[int]:
    """Filter the entire dataset first and return valid indices."""
    logger.info(f"Step 1: Pre-filtering dataset with filters: {filters}")
    
    if not filters:
        all_indices = list(range(len(player_metadata)))
        logger.info(f"No filters applied - using all {len(all_indices)} players")
        return all_indices
    
    filtered_indices = []
    filter_stats = {
        'position_rejected': 0, 'league_rejected': 0, 'age_rejected': 0,
        'value_rejected': 0, 'nationality_rejected': 0, 'performance_rejected': 0,
        'total_processed': 0
    }
    
    for idx in range(len(player_metadata)):
        try:
            filter_stats['total_processed'] += 1
            metadata = player_metadata[idx]
            detailed = detailed_profiles[idx] if idx < len(detailed_profiles) else {}
            passed_all_filters = True
            
            # Position filtering
            if 'position' in filters:
                filter_pos = filters['position'].lower()
                position_match = False
                player_sub_position = detailed.get('sub_position', '').lower() or metadata.get('sub_position', '').lower()
                
                if player_sub_position:
                    if filter_pos in player_sub_position or player_sub_position in filter_pos:
                        position_match = True
                    elif filter_pos.replace('-', ' ') in player_sub_position.replace('-', ' '):
                        position_match = True
                    elif player_sub_position.replace('-', ' ') in filter_pos.replace('-', ' '):
                        position_match = True
                    # Specific sub-position matches
                    elif filter_pos in ['striker', 'centre-forward', 'center-forward'] and 'centre-forward' in player_sub_position:
                        position_match = True
                    elif filter_pos in ['second striker', 'support striker', 'false 9'] and 'second striker' in player_sub_position:
                        position_match = True
                    elif filter_pos in ['left winger', 'left wing'] and 'left winger' in player_sub_position:
                        position_match = True
                    elif filter_pos in ['right winger', 'right wing'] and 'right winger' in player_sub_position:
                        position_match = True
                    elif filter_pos in ['attacking midfield', 'cam', 'playmaker'] and 'attacking midfield' in player_sub_position:
                        position_match = True
                    elif filter_pos in ['central midfield', 'cm'] and 'central midfield' in player_sub_position:
                        position_match = True
                    elif filter_pos in ['defensive midfield', 'cdm', 'holding midfielder'] and 'defensive midfield' in player_sub_position:
                        position_match = True
                    elif filter_pos in ['left midfielder'] and 'left midfield' in player_sub_position:
                        position_match = True
                    elif filter_pos in ['right midfielder'] and 'right midfield' in player_sub_position:
                        position_match = True
                    elif filter_pos in ['centre-back', 'center-back', 'central defender', 'cb'] and 'centre-back' in player_sub_position:
                        position_match = True
                    elif filter_pos in ['left-back', 'left back', 'lb'] and 'left-back' in player_sub_position:
                        position_match = True
                    elif filter_pos in ['right-back', 'right back', 'rb'] and 'right-back' in player_sub_position:
                        position_match = True
                    # Category matching for broad queries
                    elif filter_pos == 'forward' and any(term in player_sub_position for term in ['centre-forward', 'second striker']):
                        position_match = True
                    elif filter_pos == 'winger' and any(term in player_sub_position for term in ['left winger', 'right winger']):
                        position_match = True
                    elif filter_pos == 'midfielder' and any(term in player_sub_position for term in ['attacking midfield', 'central midfield', 'defensive midfield', 'left midfield', 'right midfield']):
                        position_match = True
                    elif filter_pos == 'defender' and any(term in player_sub_position for term in ['centre-back', 'left-back', 'right-back']):
                        position_match = True
                    elif filter_pos == 'goalkeeper' and 'goalkeeper' in player_sub_position:
                        position_match = True
                
                if not position_match:
                    filter_stats['position_rejected'] += 1
                    passed_all_filters = False
                    continue
            
            # Age filtering
            player_age = metadata.get('age', 0)
            if player_age > 0:
                if 'min_age' in filters and player_age < filters['min_age']:
                    filter_stats['age_rejected'] += 1
                    passed_all_filters = False
                    continue
                if 'max_age' in filters and player_age > filters['max_age']:
                    filter_stats['age_rejected'] += 1
                    passed_all_filters = False
                    continue
                if 'target_age' in filters and abs(player_age - filters['target_age']) > 2:
                    filter_stats['age_rejected'] += 1
                    passed_all_filters = False
                    continue
            
            # Market value filtering
            player_value = metadata.get('market_value', 0)
            if 'min_market_value' in filters and player_value < filters['min_market_value']:
                filter_stats['value_rejected'] += 1
                passed_all_filters = False
                continue
            if 'max_market_value' in filters and player_value > filters['max_market_value']:
                filter_stats['value_rejected'] += 1
                passed_all_filters = False
                continue
            
            # Nationality filtering
            if 'nationality' in filters:
                player_nat = metadata.get('nationality', '').lower()
                filter_nat = filters['nationality'].lower()
                if not (filter_nat == player_nat or filter_nat in player_nat or player_nat in filter_nat):
                    filter_stats['nationality_rejected'] += 1
                    passed_all_filters = False
                    continue
            
            # League filtering
            if 'target_leagues' in filters:
                league_match = False
                current_club = metadata.get('current_club', '').lower()
                
                # Check current club against static club-league mapping
                for club_keyword, league_code in club_league_mapping.items():
                    if club_keyword in current_club:
                        for target_league in filters['target_leagues']:
                            if target_league.lower() == league_code or target_league.lower() in league_code:
                                league_match = True
                                break
                        if league_match: break
                
                # If no club match, check detailed profile's league experience
                if not league_match:
                    career_stats = detailed.get('career_stats', {})
                    league_experience = career_stats.get('league_experience', {})
                    top_5_leagues = league_experience.get('top_5_leagues', [])
                    european_comps = league_experience.get('european_competitions', [])
                    
                    for league_info in top_5_leagues:
                        league_name = league_info.get('league', '').lower()
                        for target_league in filters['target_leagues']:
                            target_lower = target_league.lower()
                            if (target_lower in league_name or league_name in target_lower or
                                (target_lower == 'gb1' and 'premier' in league_name) or
                                (target_lower == 'es1' and 'liga' in league_name) or
                                (target_lower == 'l1' and 'bundesliga' in league_name) or
                                (target_lower == 'it1' and 'serie' in league_name) or
                                (target_lower == 'fr1' and 'ligue' in league_name)):
                                league_match = True
                                break
                        if league_match: break
                    
                    # Also consider European competitions if broad "top/european leagues" is queried
                    if not league_match and any('top' in str(tl).lower() or 'european' in str(tl).lower() for tl in filters['target_leagues']):
                        if len(european_comps) > 0 or len(top_5_leagues) > 0:
                            league_match = True
                
                if not league_match:
                    filter_stats['league_rejected'] += 1
                    passed_all_filters = False
                    continue
            
            # Performance filtering
            if any(key in filters for key in ['min_goals_per_game', 'min_assists_per_game', 'min_appearances']):
                career_stats = detailed.get('career_stats', {})
                
                if 'min_goals_per_game' in filters:
                    goals_per_game = career_stats.get('goals_per_appearance', 0)
                    if goals_per_game < filters['min_goals_per_game']:
                        filter_stats['performance_rejected'] += 1
                        passed_all_filters = False
                        continue
                
                if 'min_assists_per_game' in filters:
                    assists_per_game = career_stats.get('assists_per_appearance', 0)
                    if assists_per_game < filters['min_assists_per_game']:
                        filter_stats['performance_rejected'] += 1
                        passed_all_filters = False
                        continue
                
                if 'min_appearances' in filters:
                    appearances = career_stats.get('total_appearances', 0)
                    if appearances < filters['min_appearances']:
                        filter_stats['performance_rejected'] += 1
                        passed_all_filters = False
                        continue
            
            if passed_all_filters:
                filtered_indices.append(idx)
                
        except Exception as e:
            logger.warning(f"Error filtering player {idx}: {e}")
            continue
    
    logger.info(f"Pre-filtering results:")
    logger.info(f"  Total players processed: {filter_stats['total_processed']}")
    logger.info(f"  Players passing all filters: {len(filtered_indices)}")
    for filter_type, count in filter_stats.items():
        if count > 0 and filter_type != 'total_processed':
            logger.info(f"  {filter_type}: {count} players rejected")
    
    if len(filtered_indices) == 0:
        logger.warning("No players found with all filters! Applying fallback strategy...")
        return apply_fallback_strategies(filters, player_metadata, detailed_profiles, league_mappings)
    
    return filtered_indices

def apply_fallback_strategies(original_filters: Dict[str, Any], player_metadata: List[Dict], detailed_profiles: List[Dict], league_mappings: Dict[str, List[str]]) -> List[int]:
    """Apply fallback strategies without recursion."""
    fallback_strategies = create_fallback_strategies(original_filters)
    
    for attempt_num, fallback_filters in enumerate(fallback_strategies, 1):
        logger.info(f"Fallback attempt {attempt_num}: {fallback_filters}")
        
        fallback_indices = []
        filter_stats = { # Reset stats for each fallback attempt
            'position_rejected': 0, 'league_rejected': 0, 'age_rejected': 0,
            'value_rejected': 0, 'nationality_rejected': 0, 'performance_rejected': 0,
            'total_processed': 0
        }
        
        for idx in range(len(player_metadata)):
            try:
                filter_stats['total_processed'] += 1
                metadata = player_metadata[idx]
                detailed = detailed_profiles[idx] if idx < len(detailed_profiles) else {}
                passed_all_filters = True
                
                if 'position' in fallback_filters:
                    filter_pos = fallback_filters['position'].lower()
                    position_match = False
                    player_sub_position = detailed.get('sub_position', '').lower() or metadata.get('sub_position', '').lower()
                    
                    if player_sub_position:
                        if filter_pos in player_sub_position or player_sub_position in filter_pos:
                            position_match = True
                        elif filter_pos.replace('-', ' ') in player_sub_position.replace('-', ' '):
                            position_match = True
                        elif player_sub_position.replace('-', ' ') in filter_pos.replace('-', ' '):
                            position_match = True
                        elif filter_pos in ['striker', 'centre-forward', 'center-forward'] and 'centre-forward' in player_sub_position:
                            position_match = True
                        elif filter_pos in ['second striker', 'support striker', 'false 9'] and 'second striker' in player_sub_position:
                            position_match = True
                        elif filter_pos in ['left winger', 'left wing'] and 'left winger' in player_sub_position:
                            position_match = True
                        elif filter_pos in ['right winger', 'right wing'] and 'right winger' in player_sub_position:
                            position_match = True
                        elif filter_pos in ['attacking midfield', 'cam', 'playmaker'] and 'attacking midfield' in player_sub_position:
                            position_match = True
                        elif filter_pos in ['central midfield', 'cm'] and 'central midfield' in player_sub_position:
                            position_match = True
                        elif filter_pos in ['defensive midfield', 'cdm', 'holding midfielder'] and 'defensive midfield' in player_sub_position:
                            position_match = True
                        elif filter_pos in ['left midfielder'] and 'left midfield' in player_sub_position:
                            position_match = True
                        elif filter_pos in ['right midfielder'] and 'right midfield' in player_sub_position:
                            position_match = True
                        elif filter_pos in ['centre-back', 'center-back', 'central defender', 'cb'] and 'centre-back' in player_sub_position:
                            position_match = True
                        elif filter_pos in ['left-back', 'left back', 'lb'] and 'left-back' in player_sub_position:
                            position_match = True
                        elif filter_pos in ['right-back', 'right back', 'rb'] and 'right-back' in player_sub_position:
                            position_match = True
                        elif filter_pos == 'forward' and any(term in player_sub_position for term in ['centre-forward', 'second striker']):
                            position_match = True
                        elif filter_pos == 'winger' and any(term in player_sub_position for term in ['left winger', 'right winger']):
                            position_match = True
                        elif filter_pos == 'midfielder' and any(term in player_sub_position for term in ['attacking midfield', 'central midfield', 'defensive midfield', 'left midfield', 'right midfield']):
                            position_match = True
                        elif filter_pos == 'defender' and any(term in player_sub_position for term in ['centre-back', 'left-back', 'right-back']):
                            position_match = True
                        elif filter_pos == 'goalkeeper' and 'goalkeeper' in player_sub_position:
                            position_match = True
                    
                    if not position_match:
                        filter_stats['position_rejected'] += 1
                        passed_all_filters = False
                        continue
                
                player_age = metadata.get('age', 0)
                if player_age > 0:
                    if 'min_age' in fallback_filters and player_age < fallback_filters['min_age']:
                        filter_stats['age_rejected'] += 1
                        passed_all_filters = False
                        continue
                    if 'max_age' in fallback_filters and player_age > fallback_filters['max_age']:
                        filter_stats['age_rejected'] += 1
                        passed_all_filters = False
                        continue
                    if 'target_age' in fallback_filters and abs(player_age - fallback_filters['target_age']) > 2:
                        filter_stats['age_rejected'] += 1
                        passed_all_filters = False
                        continue
                
                player_value = metadata.get('market_value', 0)
                if 'min_market_value' in fallback_filters and player_value < fallback_filters['min_market_value']:
                    filter_stats['value_rejected'] += 1
                    passed_all_filters = False
                    continue
                if 'max_market_value' in fallback_filters and player_value > fallback_filters['max_market_value']:
                    filter_stats['value_rejected'] += 1
                    passed_all_filters = False
                    continue
                
                if 'nationality' in fallback_filters:
                    player_nat = metadata.get('nationality', '').lower()
                    filter_nat = fallback_filters['nationality'].lower()
                    if not (filter_nat == player_nat or filter_nat in player_nat or player_nat in filter_nat):
                        filter_stats['nationality_rejected'] += 1
                        passed_all_filters = False
                        continue
                
                if 'target_leagues' in fallback_filters:
                    league_match = False
                    current_club = metadata.get('current_club', '').lower()
                    for club_keyword, league_code in club_league_mapping.items():
                        if club_keyword in current_club:
                            for target_league in fallback_filters['target_leagues']:
                                if target_league.lower() == league_code or target_league.lower() in league_code:
                                    league_match = True
                                    break
                            if league_match: break
                    if not league_match:
                        career_stats = detailed.get('career_stats', {})
                        league_experience = career_stats.get('league_experience', {})
                        top_5_leagues = league_experience.get('top_5_leagues', [])
                        european_comps = league_experience.get('european_competitions', [])
                        for league_info in top_5_leagues:
                            league_name = league_info.get('league', '').lower()
                            for target_league in fallback_filters['target_leagues']:
                                target_lower = target_league.lower()
                                if (target_lower in league_name or league_name in target_lower or
                                    (target_lower == 'gb1' and 'premier' in league_name) or
                                    (target_lower == 'es1' and 'liga' in league_name) or
                                    (target_lower == 'l1' and 'bundesliga' in league_name) or
                                    (target_lower == 'it1' and 'serie' in league_name) or
                                    (target_lower == 'fr1' and 'ligue' in league_name)):
                                    league_match = True
                                    break
                            if league_match: break
                        if not league_match and any('top' in str(tl).lower() or 'european' in str(tl).lower() for tl in fallback_filters['target_leagues']):
                            if len(european_comps) > 0 or len(top_5_leagues) > 0:
                                league_match = True
                    if not league_match:
                        filter_stats['league_rejected'] += 1
                        passed_all_filters = False
                        continue
                
                if any(key in fallback_filters for key in ['min_goals_per_game', 'min_assists_per_game', 'min_appearances']):
                    career_stats = detailed.get('career_stats', {})
                    if 'min_goals_per_game' in fallback_filters:
                        goals_per_game = career_stats.get('goals_per_appearance', 0)
                        if goals_per_game < fallback_filters['min_goals_per_game']:
                            filter_stats['performance_rejected'] += 1
                            passed_all_filters = False
                            continue
                    if 'min_assists_per_game' in fallback_filters:
                        assists_per_game = career_stats.get('assists_per_appearance', 0)
                        if assists_per_game < fallback_filters['min_assists_per_game']:
                            filter_stats['performance_rejected'] += 1
                            passed_all_filters = False
                            continue
                    if 'min_appearances' in fallback_filters:
                        appearances = career_stats.get('total_appearances', 0)
                        if appearances < fallback_filters['min_appearances']:
                            filter_stats['performance_rejected'] += 1
                            passed_all_filters = False
                            continue

                if passed_all_filters:
                    fallback_indices.append(idx)
                    
            except Exception as e:
                logger.warning(f"Error in fallback filtering for player {idx}: {e}")
                continue
        
        logger.info(f"Fallback attempt {attempt_num} results: {len(fallback_indices)} players found")
        
        if len(fallback_indices) > 0:
            logger.info(f"Fallback attempt {attempt_num} successful!")
            return fallback_indices
    
    logger.warning("All fallback attempts failed! Using entire dataset.")
    return list(range(len(player_metadata)))

def vector_similarity_search_on_filtered_dataset(query: str, filtered_indices: List[int], player_embeddings: np.ndarray, embedding_model: SentenceTransformer, top_k: int = 100) -> List[Tuple[int, float]]:
    """Perform vector similarity search only on pre-filtered dataset."""
    try:
        logger.info(f"Step 2: Vector similarity search on {len(filtered_indices)} pre-filtered players...")
        
        if len(filtered_indices) == 0:
            logger.warning("No players in filtered dataset!")
            return []
        
        with torch.no_grad():
            query_embedding = embedding_model.encode(
                query,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=1,
                device='cpu',
                normalize_embeddings=True
            ).reshape(1, -1)
        
        filtered_embeddings = player_embeddings[filtered_indices]
        
        similarities = cosine_similarity(query_embedding, filtered_embeddings)[0]
        
        top_k_actual = min(top_k, len(filtered_indices))
        top_local_indices = np.argsort(similarities)[::-1][:top_k_actual]
        
        results = [
            (filtered_indices[local_idx], float(similarities[local_idx])) 
            for local_idx in top_local_indices
        ]
        
        logger.info(f"Found {len(results)} candidates from filtered dataset")
        logger.info(f"Similarity range: {min(similarities):.3f} to {max(similarities):.3f}")
        
        return results
    except Exception as e:
        logger.error(f"Filtered vector search failed: {e}")
        if filtered_indices:
            indices = np.random.choice(
                filtered_indices, 
                size=min(top_k, len(filtered_indices)), 
                replace=False
            )
            return [(int(idx), 0.5) for idx in indices]
        return []

def rerank_candidates(query: str, candidates: List[Tuple[int, float]], player_metadata: List[Dict], reranker_model: CrossEncoder, top_k: int = 30) -> List[Tuple[int, float]]:
    """Rerank candidates using cross-encoder with fallback logic."""
    try:
        logger.info(f"Step 3: Reranking {len(candidates)} candidates with cross-encoder...")
        
        if len(candidates) == 0:
            return []
        
        query_doc_pairs = []
        candidate_indices = []
        original_scores = []
        
        for idx, similarity_score in candidates:
            try:
                player_text = player_metadata[idx]['embedding_text']
                query_doc_pairs.append([query, player_text])
                candidate_indices.append(idx)
                original_scores.append(similarity_score)
            except (IndexError, KeyError) as e:
                logger.warning(f"Skipping invalid candidate {idx}: {e}")
                continue
        
        if not query_doc_pairs:
            logger.warning("No valid query-document pairs for reranking")
            return candidates[:top_k]
        
        try:
            logger.info(f"Reranking {len(query_doc_pairs)} candidates...")
            rerank_scores = reranker_model.predict(query_doc_pairs)
            
            reranked_results = [
                (candidate_indices[i], float(rerank_scores[i]))
                for i in range(len(candidate_indices))
            ]
            
            reranked_results.sort(key=lambda x: x[1], reverse=True)
            
            top_reranked = reranked_results[:top_k]
            
            logger.info(f"Reranked to top {len(top_reranked)} most relevant players")
            logger.info(f"Rerank score range: {reranked_results[-1][1]:.3f} to {reranked_results[0][1]:.3f}")
            
            return top_reranked
        except Exception as rerank_error:
            logger.error(f"Reranking failed: {rerank_error}")
            logger.warning("Falling back to similarity scores...")
            
            fallback_results = [
                (candidate_indices[i], original_scores[i])
                for i in range(len(candidate_indices))
            ]
            
            fallback_results.sort(key=lambda x: x[1], reverse=True)
            
            top_fallback = fallback_results[:top_k]
            
            logger.info(f"📋 Using top {len(top_fallback)} candidates by similarity")
            return top_fallback
            
    except Exception as e:
        logger.error(f"Complete reranking failure: {e}")
        logger.warning("Using original candidate order...")
        return candidates[:top_k]

def prepare_player_data(ranked_players: List[Tuple[int, float]], player_metadata: List[Dict], detailed_profiles: List[Dict]) -> List[Dict[str, Any]]:
    """Prepare structured player data from ranked indices."""
    logger.info(f"Step 4: Preparing player data for {len(ranked_players)} players...")
    player_data = []
    
    for rank, (idx, relevance_score) in enumerate(ranked_players, 1):
        try:
            metadata = player_metadata[idx]
            detailed = detailed_profiles[idx] if idx < len(detailed_profiles) else {}
            
            career_stats = detailed.get('career_stats', {})
            playing_style = detailed.get('playing_style', {})
            transfer_history = detailed.get('transfer_history', {})
            market_trends = detailed.get('market_value_trends', {})
            
            sub_position = detailed.get('sub_position', '') or metadata.get('sub_position', '')
            
            player_info = {
                'rank': rank,
                'relevance_score': round(relevance_score, 3),
                'basic_info': {
                    'name': metadata.get('name', 'Unknown'),
                    'age': metadata.get('age', 0),
                    'position': sub_position,
                    'sub_position': sub_position,
                    'nationality': metadata.get('nationality', 'Unknown'),
                    'current_club': metadata.get('current_club', 'Free Agent'),
                    'market_value': metadata.get('market_value', 0)
                },
                'performance': {
                    'total_appearances': career_stats.get('total_appearances', 0),
                    'total_goals': career_stats.get('total_goals', 0),
                    'total_assists': career_stats.get('total_assists', 0),
                    'goals_per_game': round(career_stats.get('goals_per_appearance', 0), 3),
                    'assists_per_game': round(career_stats.get('assists_per_appearance', 0), 3),
                    'minutes_played': career_stats.get('total_minutes', 0)
                },
                'profile': {
                    'experience_level': playing_style.get('experience_level', 'unknown'),
                    'goal_scoring_ability': playing_style.get('goal_scoring_ability', 'unknown'),
                    'discipline': playing_style.get('discipline', 'unknown'),
                    'transfer_count': transfer_history.get('total_transfers', 0),
                    'career_trajectory': transfer_history.get('career_trajectory', 'unknown')
                },
                'market_info': {
                    'peak_value': market_trends.get('peak_market_value', 0),
                    'value_trend': market_trends.get('value_trend', 'stable'),
                    'value_change': market_trends.get('recent_value_change', 0)
                },
                'scouting_summary': metadata.get('embedding_text', '')
            }
            player_data.append(player_info)
        except Exception as e:
            logger.warning(f"Error preparing data for player {idx}: {e}")
            continue
    
    logger.info(f"Prepared data for {len(player_data)} players")
    return player_data

def create_gemini_prompt(query: str, top_players: List[Dict[str, Any]]) -> str:
    """Create optimized prompt for Gemini."""
    prompt = f"""You are a professional soccer scout with 20+ years of experience analyzing players for top clubs worldwide. Provide expert analysis for this scouting request.

SCOUTING REQUEST: "{query}"

TOP RECOMMENDED PLAYERS:
"""
    for i, player in enumerate(top_players, 1):
        basic = player['basic_info']
        perf = player['performance']
        profile = player['profile']
        market = player['market_info']
        
        prompt += f"""
Player {i}: {basic['name']}
• Position: {basic['position']}{f" (detailed: {basic['sub_position']})" if basic.get('sub_position') and basic['sub_position'] != basic['position'] else ""} | Age: {basic['age']} | Nation: {basic['nationality']}
• Current Club: {basic['current_club']} | Market Value: €{basic['market_value']:,}
• Career Stats: {perf['total_goals']} goals, {perf['total_assists']} assists in {perf['total_appearances']} appearances
• Performance Ratios: {perf['goals_per_game']:.3f} goals/game, {perf['assists_per_game']:.3f} assists/game
• Player Profile: {profile['experience_level']} player, {profile['goal_scoring_ability']} goalscorer, {profile['discipline']} discipline
• Career: {profile['transfer_count']} transfers, {profile['career_trajectory']} trajectory
• Market: Peak value €{market['peak_value']:,}, trend: {market['value_trend']}
• Relevance Score: {player['relevance_score']:.3f}/1.0
"""
    prompt += f"""
REQUIRED ANALYSIS:
1. **Match Assessment** (2-3 sentences): How well do these players meet the specific criteria?
2. **Top Recommendation** (3-4 sentences): Deep dive on #1 pick - strengths, style, why they're perfect
3. **Alternative Options** (2-3 sentences): Quick analysis of players 2-3 and their unique value
4. **Key Statistics** (2 sentences): Most important performance metrics that stand out
5. **Scouting Concerns** (1-2 sentences): Any potential risks or areas to investigate
6. **Final Verdict** (1-2 sentences): Clear recommendation for the scout

Keep the analysis professional, data-driven, and actionable. Focus on insights that help make informed scouting decisions. Write in a confident, expert tone. Total length: 350-450 words.
"""
    return prompt

def generate_gemini_response(query: str, player_data: List[Dict[str, Any]], gemini_model: Optional[genai.GenerativeModel], top_n: int = 5) -> Dict[str, Any]:
    """Generate response using Gemini or fallback to template."""
    logger.info("Step 5: Generating Gemini response...")
    
    top_players = player_data[:top_n]
    
    if not gemini_model:
        logger.warning("Gemini not available - using template response")
        return generate_template_response(query, top_players)
    
    try:
        prompt = create_gemini_prompt(query, top_players)
        
        response = gemini_model.generate_content(prompt)
        ai_response = response.text
        
        logger.info("Gemini response generated successfully")
        
        return {
            'success': True,
            'query': query,
            'ai_response': ai_response,
            'top_players': top_players,
            'total_candidates_found': len(player_data),
            'search_metadata': {
                'timestamp': datetime.now().isoformat(),
                'model_used': 'gemini-1.5-flash',
                'reranking_applied': True,
                'filters_applied': True,
                'filter_first_approach': True
            }
        }
    except Exception as e:
        logger.error(f"Gemini API failed: {e}")
        # Fallback to template response if Gemini fails
        return generate_template_response(query, top_players)

def generate_template_response(query: str, top_players: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate fallback template response."""
    if not top_players:
        ai_response = f"I couldn't find any players matching your criteria: '{query}'. Try adjusting your search parameters."
    else:
        top_player = top_players[0]
        basic = top_player['basic_info']
        perf = top_player['performance']
        
        response_parts = [
            f"Based on your query '{query}', I've identified {len(top_players)} potential candidates.",
            f"My top recommendation is {basic['name']}, a {basic['age']}-year-old {basic['position']} from {basic['nationality']}.",
            f"Currently at {basic['current_club']}, they have scored {perf['total_goals']} goals in {perf['total_appearances']} appearances ({perf['goals_per_game']:.2f} per game).",
            f"With a market value of €{basic['market_value']:,}, they represent excellent value for your scouting requirements."
        ]
        
        if len(top_players) > 1:
            response_parts.append(f"Other strong candidates include {top_players[1]['basic_info']['name']} and {top_players[2]['basic_info']['name'] if len(top_players) > 2 else 'others'}, each bringing unique strengths to consider.")
        
        ai_response = " ".join(response_parts)
    
    return {
        'success': True,
        'query': query,
        'ai_response': ai_response,
        'top_players': top_players,
        'total_candidates_found': len(top_players), # Adjusted for template to reflect only top_n
        'search_metadata': {
            'timestamp': datetime.now().isoformat(),
            'model_used': 'template_fallback',
            'reranking_applied': False, # Reranking wasn't explicitly used if Gemini failed
            'filters_applied': True,
            'filter_first_approach': True
        }
    }

# --- Main RAG Search Pipeline ---
def run_search(query: str, player_embeddings: np.ndarray, player_metadata: List[Dict], detailed_profiles: List[Dict],
               embedding_model: SentenceTransformer, reranker_model: CrossEncoder, gemini_model: Optional[genai.GenerativeModel],
               league_mappings: Dict[str, List[str]], top_k_initial: int = 100, top_k_rerank: int = 30, top_n_final: int = 5) -> Dict[str, Any]:
    """
    Main search function with filter-first approach.
    All necessary data and models are passed as arguments.
    """
    logger.info(f"Starting FILTER-FIRST RAG pipeline for query: '%s'", query)
    
    try:
        # Step 1: Parse query and extract filters
        filters = parse_query(query, league_mappings)
        
        # Step 2: Pre-filter the entire dataset based on extracted filters
        filtered_indices = get_filtered_dataset_indices(filters, player_metadata, detailed_profiles, league_mappings)
        
        if len(filtered_indices) == 0:
            logger.error("No players found after filtering! Returning empty results.")
            return {
                'success': False,
                'query': query,
                'error': 'No players match the specified criteria after strict filtering.',
                'ai_response': f"No players found matching your specific criteria: '%s'. Please broaden your search or adjust filters." % query,
                'top_players': [],
                'total_candidates_found': 0,
                'search_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'filter_first_approach': True,
                    'filtered_dataset_size': 0
                }
            }
        
        # Step 3: Perform vector similarity search ONLY on filtered dataset
        vector_candidates = vector_similarity_search_on_filtered_dataset(
            query, filtered_indices, player_embeddings, embedding_model, top_k_initial
        )
        
        # Step 4: Rerank the candidates from filtered dataset
        reranked_candidates = rerank_candidates(query, vector_candidates, player_metadata, reranker_model, top_k_rerank)
        
        # Step 5: Prepare player data for LLM
        player_data = prepare_player_data(reranked_candidates, player_metadata, detailed_profiles)
        
        # Step 6: Generate final response using Gemini
        final_response = generate_gemini_response(query, player_data, gemini_model, top_n_final)
        
        # Add metadata about the filtering process
        final_response['search_metadata'].update({
            'filter_first_approach': True,
            'original_dataset_size': len(player_metadata),
            'filtered_dataset_size': len(filtered_indices),
            'filters_applied': filters,
            'filtering_efficiency': f"%d/%d players retained" % (len(filtered_indices), len(player_metadata))
        })
        
        logger.info("FILTER-FIRST RAG pipeline completed successfully!")
        logger.info(f"Dataset reduction: %d -> %d players (%s retained)" % (len(player_metadata), len(filtered_indices), final_response['search_metadata']['filtering_efficiency']))
        
        return final_response
        
    except Exception as e:
        logger.error(f"FILTER-FIRST RAG pipeline failed: %s", e)
        return {
            'success': False,
            'query': query,
            'error': str(e),
            'ai_response': f"I encountered an error processing your query: %s" % str(e),
            'top_players': [],
            'total_candidates_found': 0,
            'search_metadata': {
                'timestamp': datetime.now().isoformat(),
                'error': True,
                'filter_first_approach': True
            }
        }

# --- Streamlit App Layout ---
st.set_page_config(page_title="⚽ Soccer Scout AI Chatbot", layout="centered")

st.title("⚽ Soccer Scout AI Chatbot")
st.markdown(
    """
    Your intelligent assistant for finding the perfect football players!
    Ask me about player positions, age, nationality, leagues, market value, and more.
    """
)

# --- Load data and models using st.cache_resource to run once ---
@st.cache_resource
def load_rag_components(embeddings_dir: Path, embedding_model_name: str, reranker_model_name: str):
    logger.info("Loading RAG components (cached)...")
    
    # Data loading
    player_embeddings: Optional[np.ndarray] = None
    player_metadata: Optional[List[Dict]] = None
    detailed_profiles: Optional[List[Dict]] = None

    try:
        embeddings_file = embeddings_dir / "player_embeddings.npy"
        if embeddings_file.exists():
            player_embeddings = np.load(embeddings_file)
            logger.info(f"Loaded %d player embeddings" % player_embeddings.shape[0])
        else:
            st.error(f"Embeddings file not found: %s. Please run the `embedding_generator` script first." % embeddings_file)
            st.stop() # Stop the app if crucial files are missing
        
        metadata_file = embeddings_dir / "player_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                player_metadata = json.load(f)
            logger.info(f"Loaded metadata for %d players" % len(player_metadata))
        else:
            st.error(f"Metadata file not found: %s. Please ensure it exists." % metadata_file)
            st.stop()
        
        profiles_file = embeddings_dir / "detailed_player_profiles.json"
        if profiles_file.exists():
            with open(profiles_file, 'r', encoding='utf-8') as f:
                detailed_profiles = json.load(f)
            logger.info(f"Loaded detailed profiles for %d players" % len(detailed_profiles))
        else:
            logger.warning(f"Detailed profiles file not found at %s. Some player details might be missing in responses." % profiles_file)
            detailed_profiles = [{}] * len(player_metadata) # Initialize with empty dicts to match length
            
    except Exception as e:
        st.error(f"Failed to load data components: %s" % e)
        st.stop()
    
    # Model initialization
    embedding_model: Optional[SentenceTransformer] = None
    reranker_model: Optional[CrossEncoder] = None
    
    try:
        embedding_model = SentenceTransformer(embedding_model_name, device='cpu')
        logger.info(f"Loaded embedding model: %s" % embedding_model_name)
        
        reranker_model = CrossEncoder(reranker_model_name, device='cpu')
        logger.info(f"Loaded reranker model: %s" % reranker_model_name)
        
    except Exception as e:
        st.error(f"Failed to initialize ML models (embedding/reranker): %s. Check model names and internet connection." % e)
        # Do not stop, but set models to None so functions can handle it
        embedding_model = None
        reranker_model = None

    # Gemini setup - Read from st.secrets
    gemini_model: Optional[genai.GenerativeModel] = None
    gemini_api_key_from_secrets = st.secrets.get("GEMINI_API_KEY")
    
    if gemini_api_key_from_secrets:
        try:
            genai.configure(api_key=gemini_api_key_from_secrets)
            gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            logger.info("Gemini API configured successfully using st.secrets")
        except Exception as e:
            st.warning(f"Gemini setup failed (from st.secrets): %s. Ensure API key is valid and has access. Using template responses." % e)
            gemini_model = None
    else:
        st.warning("GEMINI_API_KEY not found in .streamlit/secrets.toml. Gemini response will be unavailable; using template responses.")
        gemini_model = None

    league_mappings = initialize_league_mappings()
    
    return player_embeddings, player_metadata, detailed_profiles, embedding_model, reranker_model, gemini_model, league_mappings

# Load components
with st.spinner("Loading AI models and player data... This might take a moment."):
    (PLAYER_EMBEDDINGS, PLAYER_METADATA, DETAILED_PROFILES, 
     EMBEDDING_MODEL, RERANKER_MODEL, GEMINI_MODEL, LEAGUE_MAPPINGS) = \
        load_rag_components(EMBEDDINGS_DIR, EMBEDDING_MODEL_NAME, RERANKER_MODEL_NAME) # Removed gemini_api_key from here

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask me about a player or scouting scenario..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating response..."):
            # Extract desired number of results from the prompt
            # Default to 5 if not specified or invalid
            requested_top_n = 5 
            match = re.search(r'(?:top|show|find)\s+(\d+)\s+players?', prompt.lower())
            if match:
                try:
                    num = int(match.group(1))
                    # Limit the requested number of players to a reasonable range (e.g., 1 to 10)
                    if 1 <= num <= 10: 
                        requested_top_n = num
                    else:
                        st.info("I can show between 1 and 10 players. Defaulting to 5.")
                except ValueError:
                    pass # Keep default if number can't be parsed

            # Call the main RAG search function with the dynamically determined top_n_final
            response = run_search(
                query=prompt,
                player_embeddings=PLAYER_EMBEDDINGS,
                player_metadata=PLAYER_METADATA,
                detailed_profiles=DETAILED_PROFILES,
                embedding_model=EMBEDDING_MODEL,
                reranker_model=RERANKER_MODEL,
                gemini_model=GEMINI_MODEL,
                league_mappings=LEAGUE_MAPPINGS,
                top_k_initial=150,
                top_k_rerank=40,
                top_n_final=requested_top_n # Use the dynamically determined value here
            )
            
            ai_response_content = response.get('ai_response', "Sorry, I couldn't process that request.")
            st.markdown(ai_response_content)
            st.session_state.messages.append({"role": "assistant", "content": ai_response_content})

            # Optionally, display some metadata for debugging/info
            if response.get('success') and response.get('top_players'):
                st.subheader("Top Recommended Players (Summary):")
                for player in response['top_players']:
                    basic = player['basic_info']
                    st.write(f"**{basic['name']}** (%s, %s) | Club: %s | MV: €%s | Relevance: %.3f" % (basic['age'], basic['position'], basic['current_club'], f"{basic['market_value']:,}", player['relevance_score']))
                
                st.markdown(f"---")
                st.caption(f"Filters Applied: %s" % response['search_metadata'].get('filters_applied', 'None'))
                st.caption(f"Dataset Reduction: %s -> %s players" % (response['search_metadata'].get('original_dataset_size', 'N/A'), response['search_metadata'].get('filtered_dataset_size', 'N/A')))
                st.caption(f"Processing Time: %.2f seconds" % response.get('processing_time_seconds', 0))