import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class ComparativeAnalysis(object):

    def __init__(self, input_filepath):
        self.metrics = []
        self.input_filepath = input_filepath
        self.result_df = pd.read_csv(self.input_filepath).dropna()

    def compute_metric_differences(self, group):
        before_adaptation_group = group[group['adaptation_state'] == 'before_adaptation']
        before_adaptation_row = {metric: None for metric in self.metrics}
        if len(before_adaptation_group) > 0:
            before_adaptation_row = before_adaptation_group.iloc[0]
        after_adaptation_group = group[group['adaptation_state'] == 'after_adaptation']
        after_adaptation_row = {metric: None for metric in self.metrics}
        if len(after_adaptation_group) > 0:
            after_adaptation_row = after_adaptation_group.iloc[0]
        differences = {}
        for metric in self.metrics:
            differences[metric] = None
            if before_adaptation_row[metric] and after_adaptation_row[metric]:
                differences[metric] = (before_adaptation_row[metric] - after_adaptation_row[metric]) / before_adaptation_row[metric]
        return pd.Series(differences)
    
    def compute_metric_averages(self, group):
        after_adaptation_group = group[group['adaptation_state'] == 'after_adaptation']
        averages = {}
        for metric in self.metrics:
            averages[metric] = None
            if metric in after_adaptation_group.columns:
                averages[metric] = after_adaptation_group[metric].mean()
        return pd.Series(averages)

    def plot_comparison(self, df, region_type, source_train_dataset, target_train_dataset, label):
        melted_df = pd.melt(df, id_vars=['training_method'], var_name='metric', value_name='value')
        plt.figure(figsize=(12, 8))
        sns.barplot(x='metric', y='value', hue='training_method', data=melted_df, palette='Set2')
        plt.title(
            f'{label} Across Test Metrics by Training Method for {region_type} regions in {source_train_dataset} to {target_train_dataset} Domain Adaptation Experiment')
        plt.xlabel('Metric')
        plt.ylabel(label)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.legend(title='Training Method', loc='upper right')
        plt.show()
        
        
    def plot_metrics_averages(self, metrics, target_region_type=None):
        return self.plot_metrics(metrics, comparison_function=self.compute_metric_averages, group_columns=['source_train_dataset', 'target_train_dataset', 'training_method', 'test_region_type'], label='Averages', target_region_type=target_region_type)

    def plot_metrics_differences(self, metrics, target_region_type=None):
        return self.plot_metrics(metrics, comparison_function=self.compute_metric_differences, group_columns=['source_train_dataset', 'target_train_dataset', 'training_method', 'test_dataset', 'test_region_type'], label='Difference Proportions', target_region_type=target_region_type)
        
    
    def plot_metrics(self, metrics, comparison_function, group_columns, label, target_region_type=None):

        assert comparison_function, 'Comparison Function must be passed!'
        assert group_columns, 'Group Columns must be passed!'
        self.metrics = metrics

        # Grouping by the specified columns and applying the comparison function
        result_change_df = self.result_df.groupby(group_columns).apply(comparison_function).reset_index()
        display(result_change_df)

        # Renaming columns for clarity
        renamed_columns = ['source_train_dataset', 'target_train_dataset', 'training_method', 'test_dataset',
                           'test_region_type']
        renamed_metrics = [f'{metric}_comparison' for metric in self.metrics]
        renamed_columns.extend(renamed_metrics)
        result_change_df.columns = renamed_columns

        # Dropping NaNs
        result_change_df = result_change_df.dropna()

        for source_train_dataset in result_change_df['source_train_dataset'].unique():
            for target_train_dataset in result_change_df['target_train_dataset'].unique():
                scenario_df = result_change_df[(result_change_df['source_train_dataset'] == source_train_dataset) & (
                            result_change_df['target_train_dataset'] == target_train_dataset)]
                if len(scenario_df) == 0:
                    continue  # skip
                scenario_df = scenario_df.drop(columns=['source_train_dataset', 'target_train_dataset'])
                for region_type in scenario_df['test_region_type'].unique():
                    if target_region_type and region_type != target_region_type:
                        continue
                    kitti_to_ddad_by_region_type_df = scenario_df[scenario_df['test_region_type'] == region_type]
                    kitti_to_ddad_by_region_type_df = kitti_to_ddad_by_region_type_df.drop(columns=['test_region_type'])
                    label_metric = lambda metric: metric.replace('test_', "").replace("_comparison", "").replace("_",
                                                                                                                " ").upper()
                    metric_labels = [label_metric(metric) for metric in renamed_metrics]
                    renamed_columns = ['training_method', 'test_dataset']
                    renamed_columns.extend(metric_labels)
                    kitti_to_ddad_by_region_type_df.columns = renamed_columns
                    kitti_to_ddad_by_region_type_melted_df = pd.melt(kitti_to_ddad_by_region_type_df,
                                                                     id_vars=['training_method', 'test_dataset'],
                                                                     var_name='metric', value_name='value')
                    pivoted_df = kitti_to_ddad_by_region_type_melted_df.pivot(index='training_method',
                                                                              columns=['test_dataset', 'metric'],
                                                                              values='value')
                    pivoted_df.columns = [f"{col[1]} ({col[0]})" for col in pivoted_df.columns]
                    pivoted_df = pivoted_df.reset_index()
                    display(pivoted_df)
                    self.plot_comparison(pivoted_df, region_type, source_train_dataset, target_train_dataset, label)
