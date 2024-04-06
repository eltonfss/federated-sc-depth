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
                differences[metric] = (before_adaptation_row[metric] - after_adaptation_row[metric]) / \
                                      before_adaptation_row[metric]
        return pd.Series(differences)

    def plot_diff_prop(self, df, region_type, source_train_dataset, target_train_dataset):
        melted_df = pd.melt(df, id_vars=['training_method'], var_name='metric', value_name='value')
        plt.figure(figsize=(12, 8))
        sns.barplot(x='metric', y='value', hue='training_method', data=melted_df, palette='Set2')
        plt.title(
            f'Difference Proportions Across Test Metrics by Training Method for {region_type} regions in {source_train_dataset} to {target_train_dataset} Domain Adaptation Experiment')
        plt.xlabel('Metric')
        plt.ylabel('Difference Proportion')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.legend(title='Training Method', loc='upper right')
        plt.show()

    def plot_metrics(self, metrics):

        self.metrics = metrics

        # Grouping by the specified columns and applying the function to compute differences
        result_change_df = self.result_df.groupby(
            ['source_train_dataset', 'target_train_dataset', 'training_method', 'test_dataset',
             'test_region_type']).apply(self.compute_metric_differences).reset_index()

        # Renaming columns for clarity
        renamed_columns = ['source_train_dataset', 'target_train_dataset', 'training_method', 'test_dataset',
                           'test_region_type']
        renamed_metrics = [f'{metric}_diff_prop' for metric in self.metrics]
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
                    kitti_to_ddad_by_region_type_df = scenario_df[scenario_df['test_region_type'] == region_type]
                    kitti_to_ddad_by_region_type_df = kitti_to_ddad_by_region_type_df.drop(columns=['test_region_type'])
                    label_metric = lambda metric: metric.replace('test_', "").replace("_diff_prop", "").replace("_",
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
                    self.plot_diff_prop(pivoted_df, region_type, source_train_dataset, target_train_dataset)
