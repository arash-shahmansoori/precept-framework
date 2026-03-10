#!/usr/bin/env python3
"""
Verify that all domains have deterministic condition keys.
Same condition key should always produce the same solution in train and test.
"""
import random
import sys
import os

import pytest

# Add src directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
src_dir = os.path.join(project_root, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


DOMAINS_TO_TEST = ['logistics', 'finance', 'booking', 'devops', 'integration', 'coding']


def get_generator_for_domain(domain: str):
    """Get the appropriate scenario generator for a domain."""
    if domain == 'logistics':
        from precept.scenario_generators.logistics import LogisticsScenarioGenerator
        return LogisticsScenarioGenerator(num_samples=10, train_ratio=0.8)
    elif domain == 'finance':
        from precept.scenario_generators.finance import FinanceScenarioGenerator
        return FinanceScenarioGenerator(num_samples=10, train_ratio=0.8)
    elif domain == 'booking':
        from precept.scenario_generators.booking import BookingScenarioGenerator
        return BookingScenarioGenerator(num_samples=10, train_ratio=0.8)
    elif domain == 'devops':
        from precept.scenario_generators.devops import DevOpsScenarioGenerator
        return DevOpsScenarioGenerator(num_samples=10, train_ratio=0.8)
    elif domain == 'integration':
        from precept.scenario_generators.integration import IntegrationScenarioGenerator
        return IntegrationScenarioGenerator(num_samples=10, train_ratio=0.8)
    elif domain == 'coding':
        from precept.scenario_generators.coding import CodingScenarioGenerator
        return CodingScenarioGenerator(num_samples=10, train_ratio=0.8)
    else:
        raise ValueError(f"Unknown domain: {domain}")


@pytest.mark.parametrize("domain", DOMAINS_TO_TEST)
def test_deterministic_condition_keys(domain: str):
    """Test that condition keys are deterministic for a domain."""
    generator = get_generator_for_domain(domain)
    
    # Generate multi-condition scenarios
    random.seed(42)
    scenarios = generator.generate_multi_condition_scenarios(
        num_training=8,
        num_test=4,
        num_conditions=5,
        test_mode='matched'
    )
    
    # Separate train and test
    train_scenarios = [s for s in scenarios if s.get('phase') == 'training']
    test_scenarios = [s for s in scenarios if s.get('phase') == 'test']
    
    assert len(train_scenarios) > 0, f"No training scenarios generated for {domain}"
    assert len(test_scenarios) > 0, f"No test scenarios generated for {domain}"
    
    # Build key -> solution mapping from training
    train_key_solutions = {}
    for s in train_scenarios:
        mc = s.get('multi_condition', {})
        key = mc.get('condition_key')
        solution = mc.get('solution')
        if key and solution:
            if key not in train_key_solutions:
                train_key_solutions[key] = solution
            else:
                # Same key should have same solution within training
                assert train_key_solutions[key] == solution, (
                    f"Inconsistent solutions for key {key} in training: "
                    f"{train_key_solutions[key]} vs {solution}"
                )
    
    assert len(train_key_solutions) > 0, f"No valid keys found in training for {domain}"
    
    # Check test scenarios match training
    for s in test_scenarios:
        mc = s.get('multi_condition', {})
        key = mc.get('condition_key')
        expected_solution = mc.get('expected_solution')
        
        if key and expected_solution:
            if key in train_key_solutions:
                assert train_key_solutions[key] == expected_solution, (
                    f"Mismatch for key {key}: "
                    f"training={train_key_solutions[key]}, test={expected_solution}"
                )


def test_all_domains_deterministic():
    """Comprehensive test that all domains have deterministic keys."""
    all_passed = True
    domain_results = {}

    for domain in DOMAINS_TO_TEST:
        try:
            generator = get_generator_for_domain(domain)
            
            # Generate multi-condition scenarios
            random.seed(42)
            scenarios = generator.generate_multi_condition_scenarios(
                num_training=8,
                num_test=4,
                num_conditions=5,
                test_mode='matched'
            )
            
            # Separate train and test
            train_scenarios = [s for s in scenarios if s.get('phase') == 'training']
            test_scenarios = [s for s in scenarios if s.get('phase') == 'test']
            
            # Build key -> solution mapping from training
            train_key_solutions = {}
            for s in train_scenarios:
                mc = s.get('multi_condition', {})
                key = mc.get('condition_key')
                solution = mc.get('solution')
                if key and solution:
                    if key not in train_key_solutions:
                        train_key_solutions[key] = solution
                    elif train_key_solutions[key] != solution:
                        all_passed = False
                        domain_results[domain] = 'FAIL - inconsistent training'
                        break
            
            if domain in domain_results:
                continue
            
            # Check test scenarios match training
            test_pass = True
            for s in test_scenarios:
                mc = s.get('multi_condition', {})
                key = mc.get('condition_key')
                expected_solution = mc.get('expected_solution')
                
                if key and expected_solution:
                    if key in train_key_solutions:
                        if train_key_solutions[key] != expected_solution:
                            test_pass = False
                            all_passed = False
                            break
            
            domain_results[domain] = 'PASS' if test_pass else 'FAIL - test mismatch'
                
        except Exception as e:
            domain_results[domain] = f'ERROR - {e}'
            all_passed = False

    assert all_passed, f"Some domains failed: {domain_results}"


if __name__ == '__main__':
    # Run as standalone script for manual verification
    print('=' * 80)
    print('COMPREHENSIVE VERIFICATION: Deterministic Condition Keys Across All Domains')
    print('=' * 80)

    random.seed(42)
    all_passed = True
    domain_results = {}

    for domain in DOMAINS_TO_TEST:
        print(f'\n--- Testing {domain.upper()} Domain ---')
        
        try:
            generator = get_generator_for_domain(domain)
            
            # Generate multi-condition scenarios
            random.seed(42)
            scenarios = generator.generate_multi_condition_scenarios(
                num_training=8,
                num_test=4,
                num_conditions=5,
                test_mode='matched'
            )
            
            # Separate train and test
            train_scenarios = [s for s in scenarios if s.get('phase') == 'training']
            test_scenarios = [s for s in scenarios if s.get('phase') == 'test']
            
            print(f'  Generated: {len(train_scenarios)} train, {len(test_scenarios)} test scenarios')
            
            # Build key -> solution mapping from training
            train_key_solutions = {}
            for s in train_scenarios:
                mc = s.get('multi_condition', {})
                key = mc.get('condition_key')
                solution = mc.get('solution')
                if key and solution:
                    if key not in train_key_solutions:
                        train_key_solutions[key] = solution
                    elif train_key_solutions[key] != solution:
                        print(f'  ✗ INCONSISTENT: Same key has different solutions in TRAINING!')
                        print(f'    Key: {key}')
                        print(f'    Solutions: {train_key_solutions[key]} vs {solution}')
                        all_passed = False
            
            print(f'  Unique keys in training: {len(train_key_solutions)}')
            for key, sol in list(train_key_solutions.items())[:3]:
                print(f'    - {key[:50]}... → {sol}')
            
            # Check test scenarios match training
            test_pass = True
            for s in test_scenarios:
                mc = s.get('multi_condition', {})
                key = mc.get('condition_key')
                expected_solution = mc.get('expected_solution')
                
                if key and expected_solution:
                    if key in train_key_solutions:
                        if train_key_solutions[key] == expected_solution:
                            pass  # Match!
                        else:
                            print(f'  ✗ MISMATCH: key={key[:40]}...')
                            print(f'    Training solution: {train_key_solutions[key]}')
                            print(f'    Test expected: {expected_solution}')
                            test_pass = False
                            all_passed = False
                    else:
                        print(f'  ⚠ Test key not in training (mode=matched): {key[:40]}...')
            
            if test_pass:
                print(f'  ✓ All test scenarios match training solutions')
                domain_results[domain] = 'PASS'
            else:
                domain_results[domain] = 'FAIL'
                
        except Exception as e:
            print(f'  ✗ ERROR: {e}')
            domain_results[domain] = 'ERROR'
            all_passed = False
            import traceback
            traceback.print_exc()

    print()
    print('=' * 80)
    print('SUMMARY')
    print('=' * 80)
    for domain, result in domain_results.items():
        status = '✓' if result == 'PASS' else '✗'
        print(f'  {status} {domain}: {result}')

    print()
    if all_passed:
        print('✓ ALL DOMAINS PASSED - Condition keys are deterministic!')
        sys.exit(0)
    else:
        print('✗ SOME DOMAINS FAILED - Issues found')
        sys.exit(1)
