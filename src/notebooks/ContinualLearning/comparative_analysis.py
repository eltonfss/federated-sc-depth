from datetime import datetime

from IPython.core.display_functions import display

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.patches as mpatches


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

    def compute_metric_averages(self, group):
        after_adaptation_group = group[group['adaptation_state'] == 'after_adaptation']
        averages = {}
        for metric in self.metrics:
            averages[metric] = None
            if metric in after_adaptation_group.columns:
                averages[metric] = after_adaptation_group[metric].mean()
        return pd.Series(averages)

    def plot_comparison(self, df, region_type, source_train_dataset, target_train_dataset, label,
                        save_as_pdf=False, legend_configs=None, colors=None, hatches=None, ):
        # Melt the DataFrame for plotting
        melted_df = pd.melt(df, id_vars=['training_method'], var_name='metric', value_name='value')

        # Set up the plot
        plt.figure(figsize=(12, 8))

        # Create the barplot with the specified colors
        sns.barplot(x='metric', y='value', hue='training_method', data=melted_df, palette=colors)

        # Apply hatches if specified
        if hatches:
            for bars, hatch in zip(plt.gca().containers, hatches):
                for bar in bars:
                    bar.set_hatch(hatch)

        # Customize the plot
        plt.xlabel('METRIC')
        ylabel = "DIFFERENCE PROPORTION" if "diff_prop" in label else label
        plt.ylabel(ylabel)
        # plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Configure legend
        if legend_configs is False:
            plt.legend().remove()
        elif legend_configs:
            plt.legend(**legend_configs)

        # Save the plot as a PDF if specified
        if save_as_pdf:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            plt.savefig(f'{label}_{source_train_dataset.lower()}_to_{target_train_dataset.lower()}.pdf')

        # Show the plot
        plt.show()

    def plot_metrics_averages(self, metrics, target_region_type=None, save_as_pdf=False, legend_configs={}, colors=None,
                              hatches=None, ):
        return self.plot_metrics(
            metrics, comparison_function=self.compute_metric_averages,
            group_columns=['source_train_dataset', 'target_train_dataset', 'training_method', 'test_region_type'],
            label='Averages', target_region_type=target_region_type, save_as_pdf=save_as_pdf,
            legend_configs=legend_configs,
            colors=colors, hatches=hatches,
        )

    def plot_metrics_differences(self, metrics, target_region_type=None, save_as_pdf=False, legend_configs={},
                                 colors=None, hatches=None, label=""):
        return self.plot_metrics(
            metrics, comparison_function=self.compute_metric_differences,
            group_columns=['source_train_dataset', 'target_train_dataset', 'training_method', 'test_dataset',
                           'test_region_type'],
            label=f'diff_prop_{label}', target_region_type=target_region_type, save_as_pdf=save_as_pdf,
            legend_configs=legend_configs,
            colors=colors, hatches=hatches,
        )

    def plot_metrics(self, metrics, comparison_function, group_columns, label, target_region_type=None,
                     save_as_pdf=False, legend_configs={}, colors=None, hatches=None, ):

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
                    # pivoted_df.columns = [f"{col[1]} ({col[0]})" for col in pivoted_df.columns]
                    pivoted_df.columns = [f"{col[1]}" for col in pivoted_df.columns]
                    pivoted_df = pivoted_df.reset_index()
                    display(pivoted_df)
                    self.plot_comparison(
                        pivoted_df, region_type, source_train_dataset, target_train_dataset,
                        label, save_as_pdf, legend_configs,
                        colors=colors, hatches=hatches,
                    )

    def normalize_metrics(self, df, target_metrics, inverse=False):
        """
        Normalize the specified metrics globally using MinMaxScaler.
        """
        scaler = MinMaxScaler()
        display(df.sort_values(by=['source_train_dataset', 'test_abs_rel']))
        if inverse:
            df[target_metrics] = 1 - scaler.fit_transform(df[target_metrics])
        else:
            df[target_metrics] = scaler.fit_transform(df[target_metrics])
        display(df.sort_values(by=['source_train_dataset', 'test_abs_rel']))
        return df

    def generate_radar_charts(self, df, target_metrics, label, target_methods=None, normalize=False, inverse=False,
                              colors=None, hatches=None):
        # Filter by target methods if specified
        if target_methods:
            df = pd.concat([df[df['training_method'] == target_method] for target_method in target_methods])

        # Rename training methods for clarity
        replace_dict = {
            'FedSCDepth(Average)': 'RFSCD',
            'FedSCDepth(Retrain)': 'ERFSCD',
            'FedSCDepth(Average&Retrain)': 'ERRFSCD',
            'FedSCDepth': 'FSCD',
            'BOFedSCDepth(Average)': 'RBOFSCD',
            'BOFedSCDepth(Retrain)': 'ERBOFSCD',
            'BOFedSCDepth(ConstrainedLoss)': 'LBOFSCD',
            'BOFedSCDepth(ConstrainedLossRetrain)': 'LERBOFSCD',
            'BOFedSCDepth(Average&Retrain)': 'ERRBOFSCD',
            'BOFedSCDepth': 'BOFSCD'
        }
        df = df.replace({'training_method': replace_dict})

        # Normalize metrics globally
        if normalize:
            df = self.normalize_metrics(df, target_metrics, inverse)

        # Create readable labels for metrics
        label_metric = lambda metric: metric.replace('test_', "").replace("_comparison", "").replace("_", " ").upper()
        label_by_target_metric = {metric: label_metric(metric) for metric in target_metrics}
        df = df.rename(columns=label_by_target_metric)
        target_metrics = list(label_by_target_metric.values())

        # Group data by source and target datasets
        combinations = df.groupby(['source_train_dataset', 'target_train_dataset'])

        # Function to create radar charts
        def create_radar_chart(categories, values, labels, title, filename, colors=None, hatches=None, ):
            num_vars = len(categories)
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            values = [v + [v[0]] for v in values]  # Close the loop
            angles += angles[:1]  # Duplicate first angle for full circle

            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

            for i, (value, label) in enumerate(zip(values, labels)):
                ax.plot(angles, value, label=label, linewidth=2, color=colors[i % len(colors)], linestyle='-',
                        marker='o')
                ax.fill(angles, value, alpha=0.25, color=colors[i % len(colors)], hatch=hatches[i % len(hatches)])

            ax.set_yticks([])
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=30, fontweight="bold")

            handles = []
            added_labels = set()
            for i, label in enumerate(labels):
                if label not in added_labels:
                    patch = mpatches.Patch(color=colors[i % len(colors)], label=label, hatch=hatches[i % len(hatches)],
                                           lw=2)
                    handles.append(patch)
                    added_labels.add(label)

            #ax.set_title(title, fontsize=16)
            ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.4), prop={'weight':'bold'}, ncol=2, fontsize=10)

            plt.tight_layout(rect=[0, 0, 1, 0.9])
            plt.savefig(filename)
            print(f"Saved radar chart: {filename}")
            plt.show()

        # Iterate through combinations of source and target datasets
        for (source, target), group in combinations:
            categories = target_metrics
            values = []
            labels = []

            for method in group['training_method'].unique():
                subset = group[group['training_method'] == method]
                avg_values = [subset[metric].mean() for metric in target_metrics]
                values.append(avg_values)
                labels.append(method)

            title = f'{label} Metrics Radar Chart for {source} -> {target}'
            filename = f"{source.lower()}_{target.lower()}_spto_{label.lower()[:3]}.pdf"

            create_radar_chart(categories, values, labels, title, filename, colors=colors, hatches=hatches)
