import time
import pandas as pd
from app import VectorSearchSystem  # Assuming previous implementation

class SearchPerformanceTester:
    def __init__(self, search_system):
        self.search_system = search_system

    def test_search_performance(self, queries):
        """
        Test search performance across multiple queries
        
        Args:
            queries (list): List of search queries to test
        
        Returns:
            dict: Performance metrics for each query
        """
        results = {}
        
        for query in queries:
            # Measure search time
            start_time = time.time()
            search_results = self.search_system.semantic_search(query)
            end_time = time.time()
            
            # Calculate search metrics
            results[query] = {
                'results_count': len(search_results),
                'search_time': end_time - start_time,
                'top_results': [
                    {
                        'name': result['name'], 
                        'sub_category': result['sub_category'], 
                        'distance': result['distance']
                    } for result in search_results
                ]
            }
        
        return results

    def generate_performance_report(self, performance_data):
        """
        Generate a detailed performance report
        
        Args:
            performance_data (dict): Performance metrics from test_search_performance
        
        Returns:
            pd.DataFrame: Performance report
        """
        report_data = []
        
        for query, metrics in performance_data.items():
            report_data.append({
                'Query': query,
                'Results Count': metrics['results_count'],
                'Search Time (s)': round(metrics['search_time'], 4),
                'Top Results': ', '.join([r['name'] for r in metrics['top_results']])
            })
        
        return pd.DataFrame(report_data)

def main():
    # Initialize search system and performance tester
    search_system = VectorSearchSystem()
    performance_tester = SearchPerformanceTester(search_system)
    
    # Define test queries
    test_queries = [
        "refreshing summer drink",
        "sweet cocktail",
        "tropical beverage",
        "non-alcoholic drink",
        "spicy mixed drink"
    ]
    
    # Run performance tests
    performance_data = performance_tester.test_search_performance(test_queries)
    
    # Generate and print performance report
    performance_report = performance_tester.generate_performance_report(performance_data)
    print(performance_report)
    
    # Optional: Save report to CSV
    performance_report.to_csv('search_performance_report.csv', index=False)

if __name__ == "__main__":
    main()