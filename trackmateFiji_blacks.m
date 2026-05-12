% =========================================================================
% TrackMate 3D Trajectory Analysis 
% =========================================================================
%
% Core functionality:
%   1. Import 3D particle trajectories exported from TrackMate XML files.
%   2. Filter trajectories by minimum duration and maximum single-step displacement.
%   3. Compute trajectory-level kinetic metrics, including total path length,
%      average velocity, maximum displacement, directionality, MSD curves, and
%      short-time MSD exponent alpha.
%   4. Classify trajectories into directed, Brownian-like, and confined motion
%      states based on the short-time alpha exponent.
%   5. Generate publication-style black-background figures, including 3D
%      trajectory maps, selected-track displacement traces, MSD plots, velocity-
%      displacement scatter plots, lipid-droplet size distributions, and metric
%      distribution plots.
%
% Notes:
%   - This script requires a TrackMate XML importer function named
%     importTrackMateTracks.m.
%   - Update xml_filepath, image_dims_pixels, and voxel_size_um before running.
%   - The short-time alpha is fitted only from the first alphaFitPoints MSD
%     points to emphasize local motion behavior.
%
% =========================================================================

function trackmateFiji_blacks()

clear; clc;
close all;

%% ------------------ Custom color scheme ------------------
% Define colors at the beginning so they can be easily customized.
COLOR_SCHEME.directed = [1 0 0];          % Red: directed motion
COLOR_SCHEME.confined = [0.35 0.7 0.9];  % Cyan-like: confined motion
COLOR_SCHEME.brownian = [0.5 0.5 0.5];   % Gray: Brownian-like motion

% Alpha thresholds used for motion-state visualization.
ALPHA_THRESHOLDS.super = 1.1;  % Directed: alpha >= 1.1
ALPHA_THRESHOLDS.sub = 0.9;    % Confined: alpha < 0.9
% =========================================================================

%% ------------------ 1. User control panel ------------------
% --- Analysis mode ---
ANALYSIS_MODE = 'auto'; % Options: 'auto' or 'manual'

% --- Initial trajectory filtering ---
minFrames = 5;        % Minimum number of frames required during import filtering.

% --- Analysis filter criteria ---
% Only trajectories satisfying all criteria below are used for downstream
% analysis and plotting.
minDurationForAnalysis = 4;   % Recommended minimum duration for robust short-time fitting.
maxStepForAnalysis = 2;       % Maximum allowed single-step displacement for analysis (um).

% --- MSD calculation parameters ---
maxLagFraction = 0.3;     % Maximum lag fraction used for plotting MSD curves.
alphaFitPoints = 4;       % Number of early MSD points used for alpha fitting.
minLagPoints = 3;         % Minimum number of lag points required for alpha fitting.
USE_FIXED_LAG = false;    % Set true to use a fixed maximum lag.
maxLagFrames = 5;         % Used only when USE_FIXED_LAG = true.

% --- Manual mode track IDs ---
% The selected IDs must exist after filtering.
% MANUAL_IDS.directed = 443;
% MANUAL_IDS.brownian = 8;
% MANUAL_IDS.confined = 279;

% Example IDs for zoomed-in visualization.
MANUAL_IDS.directed = 85;
MANUAL_IDS.brownian = 64;
MANUAL_IDS.confined = 59;

% --- File and physical parameters ---
% Replace this example path with the TrackMate XML file to be analyzed.
xml_filepath = fullfile(pwd, 'example_trackmate.xml');

% Confirm these dimensions based on the actual image acquisition settings.
% image_dims_pixels = [254, 254, 12];
% voxel_size_um = [0.133, 0.133, 1];
image_dims_pixels = [320, 320, 8];
voxel_size_um = [0.106, 0.106, 1.5];

% --- Plot style parameters ---
FONT_NAME = 'Arial'; FONT_SIZE = 36;
STYLE.FigColor = 'k'; STYLE.AxColor = 'k'; STYLE.TextColor = 'w'; STYLE.GridColor = 'w';
STYLE.GridAlpha = 0.25; STYLE.FaintColor = [0.4 0.4 0.4]; STYLE.FaintAlpha = 0.6;
STYLE.LineWidthHighlight = 3.5;
STYLE.LineWidthBackground = 1.5;
STYLE.LineWidthGlobalClass = 2.5;
STYLE.GridLineWidthSpecial = 2.0;
STYLE.GridColorSpecial = [0.2 0.2 0.2];
STYLE.BoxLineWidth = 2.5;
STYLE.LineColor = [0.2 0.8 0.8]; STYLE.FillColor = [0.2 0.6 0.8];

% --- Centered figure window layout ---
fig_width = 1200; fig_height = 900;
screen_size = get(0, 'ScreenSize');
left_pos = (screen_size(3) - fig_width) / 2; bottom_pos = (screen_size(4) - fig_height) / 2;
figPosition = [left_pos, bottom_pos, fig_width, fig_height];
FIG_PROPS = {'NumberTitle','off','Color',STYLE.FigColor, 'Position', figPosition, 'PaperPositionMode', 'auto'};
% =========================================================================

%% ------------------ 2. Data import, metric calculation, and filtering ------------------
[tracks, spots, info] = importTrackMateTracks(xml_filepath);
if isempty(tracks), error('No trajectories were imported from the XML file.'); end
t_interval = info.timeInverval;
physical_limits = image_dims_pixels .* voxel_size_um;
fprintf('Data import completed. Imported %d trajectories.\n', numel(tracks));

% Step 1: Filter short trajectories.
fprintf('Filtering short trajectories with minimum frame number = %d...\n', minFrames);
valid_tracks_mask = cellfun(@(t) size(t, 1) >= minFrames, tracks);
tracks = tracks(valid_tracks_mask);
fprintf('Retained %d trajectories after initial filtering.\n', numel(tracks));
if isempty(tracks), error('No trajectories remain after filtering. Please decrease minFrames.'); end

% Step 2: Calculate parameters for all remaining trajectories.
results = table();
for i = 1:numel(tracks)
    track_i = tracks{i};
    x=track_i(:,1); y=track_i(:,2); z=track_i(:,3); t=track_i(:,4); duration_frames=numel(t);
    alpha=NaN; msd_individual=[]; lag_times_ind=[];

    if duration_frames > 1
        % Basic motion parameters.
        steps=sqrt(diff(x).^2+diff(y).^2+diff(z).^2); total_length=sum(steps);
        avg_speed=ifelse(t(end)-t(1)>0, total_length/(t(end)-t(1)), 0);
        displacement_over_time=sqrt((x-x(1)).^2+(y-y(1)).^2+(z-z(1)).^2);
        max_displacement=max(displacement_over_time);
        directionality=ifelse(total_length>0, max_displacement/total_length, NaN);
        max_step=ifelse(~isempty(steps), max(steps), 0);

        % --- Core MSD calculation for short-time MSD analysis ---
        if duration_frames > minLagPoints
            % 1. Calculate lag points over a longer range for MSD plotting.
            if USE_FIXED_LAG, max_lag = min(maxLagFrames, duration_frames - 1);
            else, max_lag = min(floor((duration_frames - 1) * maxLagFraction), duration_frames - 1); end

            % Ensure that enough lag points are available for fitting.
            max_lag = max(max_lag, alphaFitPoints);
            % Prevent the lag range from exceeding the actual trajectory length.
            max_lag = min(max_lag, duration_frames - 1);

            msd_individual=zeros(max_lag, 1);
            for lag=1:max_lag
                diffs = track_i(1+lag:end, 1:3) - track_i(1:end-lag, 1:3);
                msd_individual(lag) = mean(sum(diffs.^2, 2));
            end
            lag_times_ind=(1:max_lag)'*t_interval;

            % 2. Fit alpha using only the first alphaFitPoints MSD points.
            % Check that enough valid data points are available for fitting.
            if max_lag >= minLagPoints && ~any(isnan(msd_individual)) && ~any(isinf(msd_individual)) && all(msd_individual>0)

                % Use the smaller value between calculated max_lag and
                % user-defined alphaFitPoints to avoid errors in short tracks.
                fit_limit = min(max_lag, alphaFitPoints);

                if fit_limit >= 3 % At least three points are required for a line fit.
                    % Use only the first fit_limit points for log-log fitting.
                    log_t = log(lag_times_ind(1:fit_limit));
                    log_msd = log(msd_individual(1:fit_limit));

                    p = polyfit(log_t, log_msd, 1);
                    alpha = p(1);
                end
            end
        end
    else, total_length=0; avg_speed=0; displacement_over_time=0; max_displacement=0; directionality=NaN; max_step=0;
    end
    results(i, :) = {i, duration_frames, total_length, avg_speed, max_displacement, directionality, alpha, max_step, {t}, {displacement_over_time}, {[x,y,z]}, {msd_individual}, {lag_times_ind}};
end
results.Properties.VariableNames = {'ID','DurationFrames','TotalLength','AvgSpeed','MaxDisplacement','Directionality','Alpha','MaxStep','TimeVector','DisplacementVector','Coords','MSD','LagTimes'};
fprintf('Trajectory-parameter calculation completed.\n');

% Step 3: Apply unified analysis filters.
fprintf('Filtering trajectories for analysis: duration > %d frames and maximum step < %.2f um...\n', minDurationForAnalysis, maxStepForAnalysis);
original_track_count = height(results);
analysis_mask = results.DurationFrames > minDurationForAnalysis & results.MaxStep < maxStepForAnalysis;
results = results(analysis_mask, :);
fprintf('Retained %d / %d trajectories for downstream analysis and plotting.\n', height(results), original_track_count);
if height(results) < 3, error('Fewer than three trajectories remain after filtering. Please relax the filtering parameters.'); end

%% ------------------ 3. Representative track selection ------------------
if strcmpi(ANALYSIS_MODE, 'auto')
    fprintf('Automatically selecting representative trajectories...\n');

    % Directed: Prefer tracks with alpha > 1.1 and the highest directionality.
    directed_candidates_idx = find(results.Alpha > 1.1);
    if isempty(directed_candidates_idx)
        fprintf('  Note: No trajectory with alpha > 1.1 was found. Trying alpha > 1.0...\n');
        directed_candidates_idx = find(results.Alpha > 1.0);
    end
    if isempty(directed_candidates_idx)
        fprintf('  Note: No trajectory with alpha > 1.0 was found. Selecting the trajectory with the highest alpha as directed.\n');
        [~, directed_idx] = max(results.Alpha);
    else
        [~, max_dir_idx] = max(results.Directionality(directed_candidates_idx));
        directed_idx = directed_candidates_idx(max_dir_idx);
    end

    % Brownian-like: Select a trajectory with alpha close to 1 and relatively low directionality.
    brownian_candidates_idx = find(results.Directionality < 0.7);
    if isempty(brownian_candidates_idx), brownian_candidates_idx = (1:height(results))'; end
    [~, min_abs_idx] = min(abs(results.Alpha(brownian_candidates_idx) - 1.0));
    brownian_idx = brownian_candidates_idx(min_abs_idx);

    % Confined: Prefer tracks with low alpha and small maximum displacement.
    confined_candidates_idx = find(results.Alpha < 0.5 & results.MaxDisplacement < 1.0);
    if isempty(confined_candidates_idx)
        fprintf('  Warning: No trajectory satisfying the confined-motion criteria was found. Selecting the trajectory with the lowest alpha.\n');
        [~, confined_idx] = min(results.Alpha);
    else
        [~, min_alpha_idx] = min(results.Alpha(confined_candidates_idx));
        confined_idx = confined_candidates_idx(min_alpha_idx);
    end

elseif strcmpi(ANALYSIS_MODE, 'manual')
    fprintf('Using manually specified track IDs.\n');
    directed_idx = find(results.ID == MANUAL_IDS.directed, 1);
    brownian_idx = find(results.ID == MANUAL_IDS.brownian, 1);
    confined_idx = find(results.ID == MANUAL_IDS.confined, 1);
    if isempty(directed_idx) || isempty(brownian_idx) || isempty(confined_idx)
        error('One or more manually specified IDs were not found after filtering. Please check the IDs or relax the filtering criteria.');
    end
else
    error("ANALYSIS_MODE must be either 'auto' or 'manual'.");
end

% Extract original IDs for display.
directed_id_orig = results.ID(directed_idx);
brownian_id_orig = results.ID(brownian_idx);
confined_id_orig = results.ID(confined_idx);

fprintf('Selected representative trajectories for short-time MSD analysis:\n  - Directed (ID %d, alpha = %.2f)\n  - Brownian-like (ID %d, alpha = %.2f)\n  - Confined (ID %d, alpha = %.2f)\n', ...
    directed_id_orig, results.Alpha(directed_idx), brownian_id_orig, results.Alpha(brownian_idx), confined_id_orig, results.Alpha(confined_idx));

%% ------------------ 4. Plotting based on filtered trajectories ------------------
% --- Figure 1: Interactive 3D track browser ---
figure(FIG_PROPS{:}, 'Name', 'Interactive 3D Track Browser (Filtered)');
ax1 = gca; hold(ax1, 'on'); track_plots = gobjects(height(results), 1);
for i = 1:height(results)
    track_plots(i) = plot3(ax1, results.Coords{i}(:,1), results.Coords{i}(:,2), results.Coords{i}(:,3), '-', 'Color', STYLE.FaintColor, 'LineWidth', STYLE.LineWidthBackground);
    track_plots(i).ButtonDownFcn = @(~,~) highlightTrack(i); track_plots(i).PickableParts = 'visible';
end
view(3); grid off; axis equal; xlabel('X (um)'); ylabel('Y (um)'); zlabel('Z (um)'); xlim([0 physical_limits(1)]); ylim([0 physical_limits(2)]); zlim([0 physical_limits(3)]);
plot_style(ax1, 'FontSize', 24); lgd1 = legend(ax1, track_plots(1), {'Click on a track...'}, 'Location', 'northeast'); legend_style(lgd1, 'FontSize', 24);

% --- Figure 2: Global 3D trajectories classified by alpha exponent ---
figure(FIG_PROPS{:}, 'Name', '3D Trajectories Classified by Alpha (Short-Time)');
ax2 = gca; hold(ax2, 'on');
sub_alpha_limit = ALPHA_THRESHOLDS.sub;
super_alpha_limit = ALPHA_THRESHOLDS.super;
p_super = plot3(nan,nan,nan, 'Color', COLOR_SCHEME.directed, 'LineWidth', STYLE.LineWidthGlobalClass);
p_normal = plot3(nan,nan,nan, 'Color', COLOR_SCHEME.brownian, 'LineWidth', STYLE.LineWidthGlobalClass);
p_sub = plot3(nan,nan,nan, 'Color', COLOR_SCHEME.confined, 'LineWidth', STYLE.LineWidthGlobalClass);
for i = 1:height(results)
    alpha_val = results.Alpha(i);
    if isnan(alpha_val), plot_color = STYLE.FaintColor;
    elseif alpha_val >= super_alpha_limit, plot_color = COLOR_SCHEME.directed;
    elseif alpha_val < sub_alpha_limit, plot_color = COLOR_SCHEME.confined;
    else, plot_color = COLOR_SCHEME.brownian; end
    plot3(ax2, results.Coords{i}(:,1), results.Coords{i}(:,2), results.Coords{i}(:,3), '-', 'Color', plot_color, 'LineWidth', STYLE.LineWidthGlobalClass);
end
view(3); grid off; axis equal; xlabel('X (um)'); ylabel('Y (um)'); zlabel('Z (um)'); xlim([0 physical_limits(1)]); ylim([0 physical_limits(2)]); zlim([0 physical_limits(3)]);
plot_style(ax2);
lgd = legend([p_super, p_normal, p_sub], {sprintf('Directed (alpha > %.1f)', super_alpha_limit), 'Brownian-like', sprintf('Confined (alpha < %.1f)', sub_alpha_limit)});
legend_style(lgd);

% --- Figure 3: Highlighted selected 3D trajectories ---
figure(FIG_PROPS{:}, 'Name', 'Highlighted 3D Trajectories (Filtered)');
ax3 = gca; hold(ax3, 'on');
for i = 1:height(results), plot3(ax3, results.Coords{i}(:,1), results.Coords{i}(:,2), results.Coords{i}(:,3), '-', 'Color', 'w', 'LineWidth', STYLE.LineWidthBackground); end
p1=plot3(ax3, results.Coords{directed_idx}(:,1), results.Coords{directed_idx}(:,2), results.Coords{directed_idx}(:,3), '-', 'Color', COLOR_SCHEME.directed, 'LineWidth',STYLE.LineWidthHighlight);
p2=plot3(ax3, results.Coords{brownian_idx}(:,1), results.Coords{brownian_idx}(:,2), results.Coords{brownian_idx}(:,3), '-', 'Color', COLOR_SCHEME.brownian, 'LineWidth',STYLE.LineWidthHighlight);
p3=plot3(ax3, results.Coords{confined_idx}(:,1), results.Coords{confined_idx}(:,2), results.Coords{confined_idx}(:,3), '-', 'Color', COLOR_SCHEME.confined, 'LineWidth',STYLE.LineWidthHighlight);
hold(ax3, 'off'); view(3); grid off; axis equal; xlabel('X (um)'); ylabel('Y (um)'); zlabel('Z (um)'); xlim([0 physical_limits(1)]); ylim([0 physical_limits(2)]); zlim([0 physical_limits(3)]);
plot_style(ax3);
set(ax3, 'GridColor', STYLE.GridColorSpecial, 'GridAlpha', 1, 'LineWidth', STYLE.GridLineWidthSpecial);
lgd=legend([p1,p2,p3], {sprintf('Directed (ID %d)', directed_id_orig), sprintf('Brownian-like (ID %d)', brownian_id_orig), sprintf('Confined (ID %d)', confined_id_orig)}); legend_style(lgd);

% --- Figure 4: Displacement versus time for selected trajectories ---
figure(FIG_PROPS{:}, 'Name', 'Displacement vs. Time (Selected)');
ax4 = gca; hold(ax4, 'on');
plot(ax4, results.TimeVector{directed_idx} - results.TimeVector{directed_idx}(1), results.DisplacementVector{directed_idx}, '-o','Color',COLOR_SCHEME.directed,'LineWidth',STYLE.LineWidthHighlight,'MarkerSize',12,'MarkerFaceColor',COLOR_SCHEME.directed);
plot(ax4, results.TimeVector{brownian_idx} - results.TimeVector{brownian_idx}(1), results.DisplacementVector{brownian_idx}, '-s','Color',COLOR_SCHEME.brownian,'LineWidth',STYLE.LineWidthHighlight,'MarkerSize',12,'MarkerFaceColor',COLOR_SCHEME.brownian);
plot(ax4, results.TimeVector{confined_idx} - results.TimeVector{confined_idx}(1), results.DisplacementVector{confined_idx}, '-d','Color',COLOR_SCHEME.confined,'LineWidth',STYLE.LineWidthHighlight,'MarkerSize',12,'MarkerFaceColor',COLOR_SCHEME.confined);
hold(ax4, 'off'); xlabel('Elapsed Time (s)'); ylabel('Displacement (\mum)'); grid off; plot_style(ax4);xlim([0 75]);
lgd=legend({'Directed', 'Brownian-like', 'Confined'}); legend_style(lgd);

% --- Figures 5 and 6: Individual MSD curves for selected trajectories ---
figure(FIG_PROPS{:}, 'Name', 'Individual MSDs (Linear)');
ax5 = gca; hold(ax5, 'on');
plot_msd(ax5, directed_idx, 'o-', COLOR_SCHEME.directed);
plot_msd(ax5, brownian_idx, 's-', COLOR_SCHEME.brownian);
plot_msd(ax5, confined_idx, 'd-', COLOR_SCHEME.confined);
hold(ax5, 'off'); xlabel('Lag Time \tau (s)'); ylabel('MSD (\mum^2)'); grid off; axis tight; plot_style(ax5);xlim([0 150]);
legend_style(legend({sprintf('Directed (alpha = %.2f)', results.Alpha(directed_idx)), sprintf('Brownian-like (alpha = %.2f)', results.Alpha(brownian_idx)), sprintf('Confined (alpha = %.2f)', results.Alpha(confined_idx))}));

figure(FIG_PROPS{:}, 'Name', 'Individual MSDs (Log-Log)');
ax6 = gca; hold(ax6, 'on');
plot_msd(ax6, directed_idx, 'o-', COLOR_SCHEME.directed);
plot_msd(ax6, brownian_idx, 's-', COLOR_SCHEME.brownian);
plot_msd(ax6, confined_idx, 'd-', COLOR_SCHEME.confined);
set(ax6, 'XScale', 'log', 'YScale', 'log');
hold(ax6, 'off'); xlabel('Lag Time \tau (s)'); ylabel('MSD (\mum^2)'); grid off; axis tight; plot_style(ax6);xlim([0 60]);
% Add a visual cue for the short-time fitting region.
xl = xlim(ax6); yl = ylim(ax6);
patch(ax6, [xl(1) t_interval*alphaFitPoints t_interval*alphaFitPoints xl(1)], [yl(1) yl(1) yl(2) yl(2)], 'w', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
text(ax6, t_interval*1.5, yl(2)*0.5, 'Fitting Region', 'Color', 'w', 'FontSize', 20);
legend_style(legend({'Directed', 'Brownian-like', 'Confined'}));

% --- Figure 7: Velocity-displacement scatter plot ---
figure(FIG_PROPS{:}, 'Name', 'Displacement vs. Velocity (Filtered)');
ax7 = gca; hold(ax7, 'on');
scatter(ax7, results.AvgSpeed, results.MaxDisplacement, 150, 'filled', 'MarkerFaceColor',STYLE.FaintColor, 'MarkerFaceAlpha',STYLE.FaintAlpha);
s1=scatter(ax7, results.AvgSpeed(directed_idx), results.MaxDisplacement(directed_idx), 600, COLOR_SCHEME.directed,'o','filled','MarkerEdgeColor','w');
s2=scatter(ax7, results.AvgSpeed(brownian_idx), results.MaxDisplacement(brownian_idx), 600, COLOR_SCHEME.brownian,'s','filled','MarkerEdgeColor','w');
s3=scatter(ax7, results.AvgSpeed(confined_idx), results.MaxDisplacement(confined_idx), 600, COLOR_SCHEME.confined,'d','filled','MarkerEdgeColor','w');
hold(ax7, 'off'); xlabel('Average Velocity (um/s)'); ylabel('Maximum Displacement (um)'); grid off; plot_style(ax7);
legend_style(legend([s1,s2,s3], {sprintf('Directed (ID %d)', directed_id_orig), sprintf('Brownian-like (ID %d)', brownian_id_orig), sprintf('Confined (ID %d)', confined_id_orig)}));

% --- Figure 8: Lipid-droplet radius distribution ---
figure(FIG_PROPS{:}, 'Name', 'Size Distribution');
ax8 = gca; all_radii = spots.all(:, spots.FEATURES('RADIUS')); min_radius_threshold = 0.5;
filtered_radii = all_radii(all_radii > min_radius_threshold);
histogram(ax8, filtered_radii, 50, 'FaceColor', 'y', 'EdgeColor','none');
xlabel('Lipid Droplet Radius (\mum)'); ylabel('Counts'); grid off; plot_style(ax8);

% --- Figure 9: Parameter distributions ---
figure('Name', 'Parameter Distributions (Filtered)', FIG_PROPS{:});
fprintf('\n========== Distribution statistics based on %d filtered trajectories ==========\n', height(results));
speed_data = results.AvgSpeed;
displacement_data = results.MaxDisplacement;
alpha_data = results.Alpha(~isnan(results.Alpha));
subplot(1,3,1); plot_distribution(gca, speed_data, 'Average Speed (um/s)', 'Speed Distribution', STYLE.LineColor, STYLE.FillColor);
subplot(1,3,2); plot_distribution(gca, displacement_data, 'Maximum Displacement (um)', 'Displacement Distribution', [0.8 0.2 0.4], [0.8 0.4 0.6]);
subplot(1,3,3); plot_distribution(gca, alpha_data, 'Short-Time Alpha', 'Alpha Distribution (Short-Time)', [0.4 0.6 0.2], [0.6 0.8 0.4]);
fprintf('Average velocity: %.4f +/- %.4f um/s (median: %.4f)\n', mean(speed_data, 'omitnan'), std(speed_data, 'omitnan'), median(speed_data, 'omitnan'));
fprintf('Maximum displacement: %.4f +/- %.4f um\n', mean(displacement_data, 'omitnan'), std(displacement_data, 'omitnan'));
fprintf('Alpha exponent: %.4f +/- %.4f (N = %d)\n', mean(alpha_data, 'omitnan'), std(alpha_data, 'omitnan'), numel(alpha_data));
fprintf('==========================================================================\n');

%% ------------------ Helper functions ------------------
    function out = ifelse(condition, true_val, false_val)
        if condition, out = true_val; else, out = false_val; end
    end

    function highlightTrack(track_table_idx)
        for k = 1:numel(track_plots)
            if isgraphics(track_plots(k)), track_plots(k).Color = STYLE.FaintColor; track_plots(k).LineWidth = STYLE.LineWidthBackground; end
        end
        track_plots(track_table_idx).Color = 'y'; track_plots(track_table_idx).LineWidth = STYLE.LineWidthHighlight; uistack(track_plots(track_table_idx), 'top');
        new_legend_string = sprintf('ID %d, alpha = %.2f, directionality = %.2f', results.ID(track_table_idx), results.Alpha(track_table_idx), results.Directionality(track_table_idx));
        legend(ax1, track_plots(track_table_idx), {new_legend_string}, 'Location', 'northeast');
    end

    function plot_msd(ax, track_table_idx, line_style, color)
        lags = results.LagTimes{track_table_idx};
        msd = results.MSD{track_table_idx};
        if ~isempty(lags)
            % Plot the full MSD curve.
            plot(ax, lags, msd, line_style, 'Color', color, 'LineWidth',STYLE.LineWidthHighlight,'MarkerSize',12,'MarkerFaceColor',color);
            % Mark the data points used for alpha fitting.
            hold(ax, 'on');
            fit_lim = min(length(lags), alphaFitPoints);
            plot(ax, lags(1:fit_lim), msd(1:fit_lim), 'o', 'Color', 'w', 'MarkerSize', 6, 'MarkerFaceColor', 'w');
        end
    end

    function plot_distribution(ax, data, xlabel_text, title_text, line_color, fill_color)
        hold(ax, 'on');
        if numel(data) > 1
            [f, xi] = ksdensity(data);
            fill(ax, xi, f, fill_color, 'FaceAlpha', 0.5, 'EdgeColor', 'none');
            plot(ax, xi, f, 'Color', line_color, 'LineWidth', 3);
            yl = ylim(ax);
            mean_val = mean(data, 'omitnan');
            plot(ax, [mean_val, mean_val], yl, '--', 'Color', [1 0.8 0.2], 'LineWidth', 2.5);
            text(ax, mean_val, yl(2)*0.9, sprintf('Mean: %.3f', mean_val), ...
                'Color', [1 0.8 0.2], 'FontSize', FONT_SIZE-8, 'FontName', FONT_NAME, ...
                'HorizontalAlignment', 'center', 'FontWeight', 'bold');
        end
        xlabel(xlabel_text); ylabel('Density'); title(title_text);
        plot_style(ax);
    end

    function plot_style(ax, varargin)
        p = inputParser; addParameter(p, 'FontSize', FONT_SIZE); parse(p, varargin{:}); font_size = p.Results.FontSize;
        set(ax, 'Color',STYLE.AxColor, 'XColor',STYLE.TextColor, 'YColor',STYLE.TextColor, 'ZColor',STYLE.TextColor, ...
            'GridColor',STYLE.GridColor, 'GridAlpha',STYLE.GridAlpha, 'FontName',FONT_NAME, 'FontSize',font_size, 'Box', 'on', 'LineWidth', 2);
        if isprop(ax, 'XLabel'), ax.XLabel.Color = STYLE.TextColor; end
        if isprop(ax, 'YLabel'), ax.YLabel.Color = STYLE.TextColor; end
        if isprop(ax, 'ZLabel'), ax.ZLabel.Color = STYLE.TextColor; end
        if isprop(ax, 'Title'), ax.Title.Color = STYLE.TextColor; end
    end

    function legend_style(lgd, varargin)
        p = inputParser; addParameter(p, 'FontSize', FONT_SIZE); parse(p, varargin{:}); font_size = p.Results.FontSize;
        set(lgd, 'TextColor',STYLE.TextColor,'Color',STYLE.AxColor,'EdgeColor',STYLE.TextColor,'FontSize',font_size,'FontName',FONT_NAME);
    end
end
