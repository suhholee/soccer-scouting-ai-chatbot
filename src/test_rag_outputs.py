import json
from datetime import datetime
from pathlib import Path

# Import your production RAG system
from rag_system import create_search_output

def get_test_queries():
    """Test queries across different categories and complexities"""
    return [
        # Basic position + age queries
        {
            "query": "young strikers under 25",
            "category": "Basic Position + Age",
            "description": "Simple position and age filter"
        },
        {
            "query": "experienced goalkeepers over 30",
            "category": "Basic Position + Age", 
            "description": "Veteran players in specific position"
        },
        
        # League-specific queries
        {
            "query": "wingers from Premier League",
            "category": "League-Specific",
            "description": "Position + specific league"
        },
        {
            "query": "midfielders from top 5 european leagues",
            "category": "League-Specific",
            "description": "Position + top leagues group"
        },
        
        # Nationality + position queries
        {
            "query": "brazilian attacking midfielders",
            "category": "Nationality + Position",
            "description": "Specific nationality and position"
        },
        {
            "query": "spanish defenders in La Liga",
            "category": "Nationality + Position + League",
            "description": "Multiple filters combined"
        },
        
        # Budget/value-based queries
        {
            "query": "cheap strikers under 15 million euros",
            "category": "Budget-Based",
            "description": "Market value constraint"
        },
        {
            "query": "bargain defenders from Serie A",
            "category": "Budget-Based + League",
            "description": "Budget + league combination"
        },
        
        # Performance-based queries
        {
            "query": "prolific goalscorers with high goal rates",
            "category": "Performance-Based",
            "description": "Performance metrics focus"
        },
        {
            "query": "creative playmakers from Bundesliga",
            "category": "Performance-Based + League",
            "description": "Playing style + league"
        },
        
        # Complex multi-criteria queries
        {
            "query": "young brazilian wingers under 23 from top european leagues",
            "category": "Complex Multi-Criteria",
            "description": "Age + nationality + position + league"
        },
        {
            "query": "fast left-footed wingers from England under 30m",
            "category": "Complex Multi-Criteria",
            "description": "Attributes + nationality + budget"
        },
        
        # Edge cases
        {
            "query": "versatile players who can play multiple positions",
            "category": "Edge Case",
            "description": "Ambiguous position requirement"
        },
        {
            "query": "players with contracts expiring soon",
            "category": "Edge Case",
            "description": "Contract status query"
        }
    ]

def display_test_result(test_info, result, test_number):
    """Display test result in a clear, readable format"""
    
    print(f"\n{'='*80}")
    print(f"TEST #{test_number}: {test_info['category']}")
    print(f"{'='*80}")
    print(f"Query: '{test_info['query']}'")
    print(f"Description: {test_info['description']}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not result.get('success', False):
        print(f"\nFAILED: {result.get('error', 'Unknown error')}")
        return
    
    print(f"\nSUCCESS")
    print(f"Processing Time: {result.get('processing_time_seconds', 0):.2f}s")
    print(f"Total Candidates Found: {result.get('total_candidates_found', 0)}")
    
    # AI Response
    ai_response = result.get('ai_response', '')
    print(f"\n🤖 AI RESPONSE:")
    print(f"{'-'*50}")
    if ai_response:
        # Wrap text for better readability
        words = ai_response.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > 75:  # 75 chars per line
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
            else:
                current_line.append(word)
                current_length += len(word) + 1
        
        if current_line:
            lines.append(' '.join(current_line))
        
        for line in lines:
            print(line)
    else:
        print("No AI response generated")
    
    # Top Players
    top_players = result.get('top_players', [])
    print(f"\n⭐ TOP {len(top_players)} PLAYERS:")
    print(f"{'-'*50}")
    
    for i, player in enumerate(top_players, 1):
        basic = player.get('basic_info', {})
        perf = player.get('performance', {})
        market = player.get('market_info', {})
        
        print(f"\n{i}. {basic.get('name', 'Unknown')}")
        print(f"   Position: {basic.get('position', 'N/A')} | Age: {basic.get('age', 'N/A')} | Nationality: {basic.get('nationality', 'N/A')}")
        print(f"   Club: {basic.get('current_club', 'N/A')}")
        print(f"   Market Value: €{basic.get('market_value', 0):,}")
        print(f"   Career: {perf.get('total_goals', 0)} goals, {perf.get('total_assists', 0)} assists in {perf.get('total_appearances', 0)} apps")
        print(f"   Rates: {perf.get('goals_per_game', 0):.3f} goals/game, {perf.get('assists_per_game', 0):.3f} assists/game")
        print(f"   Relevance Score: {player.get('relevance_score', 0):.3f}")
    
    # Search Metadata
    metadata = result.get('search_metadata', {})
    print(f"\nSEARCH METADATA:")
    print(f"{'-'*30}")
    print(f"Model Used: {metadata.get('model_used', 'N/A')}")
    print(f"Reranking Applied: {metadata.get('reranking_applied', 'N/A')}")
    print(f"Filters Applied: {metadata.get('filters_applied', 'N/A')}")

def run_test_suite(embeddings_dir="../embeddings", gemini_api_key=None, save_to_file=True):
    """Run all test queries and display results"""
    
    print("RAG SYSTEM OUTPUT TESTING")
    print("=" * 80)
    print("This will run various test queries and display outputs for review.")
    print("You can copy any problematic outputs to share for debugging.")
    print("=" * 80)
    
    test_queries = get_test_queries()
    all_results = []
    
    for i, test_info in enumerate(test_queries, 1):
        try:
            # Run the test
            result = create_search_output(
                query=test_info['query'],
                top_n=5,
                embeddings_dir=embeddings_dir,
                gemini_api_key=gemini_api_key
            )
            
            # Display result
            display_test_result(test_info, result, i)
            
            # Store for saving
            all_results.append({
                'test_info': test_info,
                'result': result,
                'test_number': i
            })
            
            # Wait for user input between tests (optional)
            if i < len(test_queries):
                input("\nPress Enter to continue to next test...")
                
        except Exception as e:
            print(f"\nTEST #{i} CRASHED: {e}")
            all_results.append({
                'test_info': test_info,
                'result': {'success': False, 'error': str(e)},
                'test_number': i
            })
    
    # Save all results to file
    if save_to_file:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"../outputs/rag_test_outputs_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'test_session': {
                    'timestamp': datetime.now().isoformat(),
                    'total_tests': len(test_queries),
                    'embeddings_dir': embeddings_dir,
                    'gemini_api_key_provided': gemini_api_key is not None
                },
                'test_results': all_results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\nAll test results saved to: {output_file}")
    
    # Summary
    successful_tests = sum(1 for result in all_results if result['result'].get('success', False))
    print(f"\nSUMMARY:")
    print(f"Total Tests: {len(test_queries)}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {len(test_queries) - successful_tests}")
    print(f"Success Rate: {successful_tests/len(test_queries)*100:.1f}%")

def run_single_query_test(query, category="Custom", description="User provided query"):
    """Test a single custom query"""
    
    print(f"\n{'='*80}")
    print(f"SINGLE QUERY TEST")
    print(f"{'='*80}")
    
    test_info = {
        'query': query,
        'category': category,
        'description': description
    }
    
    try:
        result = create_search_output(
            query=query,
            top_n=5,
            embeddings_dir="../embeddings",
            gemini_api_key="AIzaSyBV5xWnoAt6JltQasAvpU14nPw-RcAYDpI"
        )
        
        display_test_result(test_info, result, 1)
        
        return result
        
    except Exception as e:
        print(f"QUERY FAILED: {e}")
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    # Configuration
    EMBEDDINGS_DIR = "../embeddings"
    GEMINI_API_KEY = "AIzaSyBV5xWnoAt6JltQasAvpU14nPw-RcAYDpI"
    
    print("Choose testing mode:")
    print("1. Run full test suite (14 different queries)")
    print("2. Run single custom query")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        # Run full test suite
        run_test_suite(
            embeddings_dir=EMBEDDINGS_DIR,
            gemini_api_key=GEMINI_API_KEY,
            save_to_file=True
        )
    elif choice == "2":
        # Run single query
        custom_query = input("Enter your query: ").strip()
        if custom_query:
            run_single_query_test(custom_query)
        else:
            print("No query provided.")
    else:
        print("Invalid choice. Running default test query...")
        run_single_query_test("young brazilian strikers under 25 from top european leagues")