#!/usr/bin/env python3
"""
Test script to verify heterophilic datasets can be loaded correctly
"""
import torch
from util import get_dataset

def test_dataset(name):
    print(f"\n{'='*60}")
    print(f"Testing dataset: {name}")
    print(f"{'='*60}")

    try:
        # Load dataset
        dataset = get_dataset('./data', name)
        data = dataset[0]

        print(f"✓ Dataset loaded successfully")
        print(f"  - Number of nodes: {data.num_nodes}")
        print(f"  - Number of edges: {data.num_edges}")
        print(f"  - Number of features: {data.num_features}")
        print(f"  - Number of classes: {dataset.num_classes}")

        # Check if it has predefined splits
        has_train_mask = hasattr(data, 'train_mask') and data.train_mask is not None
        has_val_mask = hasattr(data, 'val_mask') and data.val_mask is not None
        has_test_mask = hasattr(data, 'test_mask') and data.test_mask is not None

        print(f"  - Has train_mask: {has_train_mask}")
        print(f"  - Has val_mask: {has_val_mask}")
        print(f"  - Has test_mask: {has_test_mask}")

        # Calculate homophily ratio (edge homophily)
        edge_index = data.edge_index
        y = data.y
        same_class = (y[edge_index[0]] == y[edge_index[1]]).sum().item()
        homophily = same_class / edge_index.shape[1]
        print(f"  - Edge homophily ratio: {homophily:.4f}")

        return True

    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    # Test new heterophilic datasets
    # heterophilic_datasets = ['Chameleon', 'Squirrel', 'Actor', 'Texas']
    heterophilic_datasets = ['Squirrel', 'Texas']
    print("Testing Heterophilic Datasets")
    print("="*60)

    results = {}
    for dataset_name in heterophilic_datasets:
        results[dataset_name] = test_dataset(dataset_name)

    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {name}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print(f"\n✓ All {len(results)} datasets loaded successfully!")
    else:
        failed = [name for name, success in results.items() if not success]
        print(f"\n✗ {len(failed)} dataset(s) failed: {', '.join(failed)}")
