function [ tracks, spots, info ] = importTrackMateTracks( file, suppress_output )
%IMPORTTRACKMATETRACKS import tracks from a TrackMate XML file.
%
%   [T, S, I] = IMPORTTRACKMATETRACKS(FILE) imports the data from the
%   TrackMate XML file FILE. The tracks are returned in the T cell array,
%   where each cell is a N x 4 double array representing a track. The array
%   has 4 columns, for XYZT coordinates. The spots are returned in the S
%   struct, which contains all the spots detected in the movie, with their
%   features. I is a struct that contains metadata about the source file.
%
%   [T, S, I] = IMPORTTRACKMATETRACKS(FILE, SUPPRESS_OUTPUT) allows
%   suppressing the output message from the function if SUPPRESS_OUTPUT is
%   true. Default is false.
%
%   The tracks are MATLAB cell arrays, where each cell is a N x 4 double
%   array. N is the number of spots in the track. The columns are organized
%   as follow:
%   X, Y, Z, T
%
%   The spots object is a struct with the following fields:
%
%   'all' is a M x F double array, where M is the total number of spots in
%   the movie, and F is the number of features for each spot, including
%   coordinates. The columns of this array contains the spot features, in
%   the same order as in the TrackMate file. To know what are the features
%   and in what order, you have to inspect the XML file, and look for the
%   <SpotFeatures> node. The features are declared there, in order.
%   For instance:
%   <SpotFeatures>
%     <Feature feature="QUALITY" name="Quality" shortname="Quality" isint="false"/>
%     <Feature feature="POSITION_X" name="X" shortname="X" isint="false"/>
%     <Feature feature="POSITION_Y" name="Y" shortname="Y" isint="false"/>
%     <Feature feature="POSITION_Z" name="Z" shortname="Z" isint="false"/>
%     ...
%   </SpotFeatures>
%   would mean that the 'all' array would have its first column for the
%   QUALITY feature, the second for the X coordinate, etc...
%
%   The 'FRAME' field is a M x 1 array containing the frame number for each
%   spot. This is 0-based.
%
%   The 'NAMES' field is a cell of strings, containing the name of the spot
%   features, in order.
%
%   The 'FEATURES' field is a containers.Map object, that maps feature
%   names (e.g. 'QUALITY') to the feature column index in the 'all' array.
%
%   The info object is a struct with the following fields:
%
%   'filePath'        the absolute path to the source file.
%   'folder'          the folder of the source file.
%   'fileName'        the name of the source file.
%   'spaceUnits'      the units for space.
%   'timeUnits'       the units for time.
%   'voxelSize'       a struct with the X, Y and Z voxel sizes.
%   'timeInverval'    the time interval between two frames.
%   'imageSize'       a struct with the width, height and depth of the
%                     images.
%   'nFrames'         the number of frames in the movie.
%
%   This importer is based on the work of Eric Keraly.
%   See https://github.com/ekeraly/spot-tracking-competition-importer
%
%   Author: Jean-Yves Tinevez <jeanyves.tinevez@gmail.com> March 2016

    if nargin < 2
        suppress_output = false;
    end

    if ~suppress_output
        fprintf('Importing TrackMate file: %s.\n', file);
    end

    %% Error checking

    if ~exist(file, 'file')
        error('MATLAB:importTrackMateTracks:FileNotFound', ...
            'The file %s could not be found.', file)
    end


    %% Open file and check if it is of the right type.

    xml_file = xmlread(file);
    root = xml_file.getDocumentElement;

    % Check that the file is of the right type.
    ok = root.getNodeName.equals('TrackMate'); %%%%%%%%%%% THIS IS THE CORRECTED LINE %%%%%%%%%%%%
    if ~ok
        error('MATLAB:importTrackMateTracks:BadXMLFile', ...
            'File does not seem to be a proper track file. Expected root node <TrackMate> but found <%s>.', root.getNodeName());
    end

    %% Import metadata

    % Image info
    node = root.getElementsByTagName('ImageData');
    if node.getLength > 0
        el = node.item(0);
        info.fileName = char(el.getAttribute('filename'));
        info.folder = char(el.getAttribute('folder'));
        info.filePath = fullfile(info.folder, info.fileName);
        info.imageSize.width = str2double(el.getAttribute('width'));
        info.imageSize.height = str2double(el.getAttribute('height'));
        info.imageSize.depth = str2double(el.getAttribute('nslices'));
        info.voxelSize.X = str2double(el.getAttribute('pixelwidth'));
        info.voxelSize.Y = str2double(el.getAttribute('pixelheight'));
        info.voxelSize.Z = str2double(el.getAttribute('voxeldepth'));
        info.timeInverval = str2double(el.getAttribute('timeinterval')); % sic
        info.spaceUnits = char(el.getAttribute('spatialunits'));
        info.timeUnits = char(el.getAttribute('timeunits'));
    else
        % Default values
        info.filePath = file;
        [info.folder, info.fileName, ext] = fileparts(file);
        info.fileName = [info.fileName ext];
        info.imageSize.width = 512;
        info.imageSize.height = 512;
        info.imageSize.depth = 1;
        info.voxelSize.X = 1;
        info.voxelSize.Y = 1;
        info.voxelSize.Z = 1;
        info.timeInverval = 1; % sic
        info.spaceUnits = 'pixel';
        info.timeUnits = 'frame';
    end


    %% Import spots

    node = root.getElementsByTagName('SpotFeatures');
    el = node.item(0);
    declarations = el.getElementsByTagName('Feature');
    n_features = declarations.getLength;
    spots.NAMES = cell(n_features, 1);
    spots.FEATURES = containers.Map();
    for i = 0 : n_features - 1
        feature_name = char(declarations.item(i).getAttribute('feature'));
        spots.NAMES{i+1} = feature_name;
        spots.FEATURES(feature_name) = i+1;
    end

    all_spots_node = root.getElementsByTagName('AllSpots');
    n_spots = str2double(all_spots_node.item(0).getAttribute('nspots'));
    spots.all = NaN(n_spots, n_features);
    spots.FRAME = NaN(n_spots, 1);

    spot_nodes = all_spots_node.item(0).getElementsByTagName('Spot');
    for i = 0 : n_spots - 1
        spot_el = spot_nodes.item(i);
        frame = str2double(spot_el.getAttribute('FRAME'));
        spot_id = str2double(spot_el.getAttribute('ID'));
        spots.FRAME(spot_id + 1) = frame;
        for j = 1 : n_features
            feature_name = spots.NAMES{j};
            val = str2double(spot_el.getAttribute(feature_name));
            spots.all(spot_id + 1, j) = val;
        end
    end
    info.nFrames = max(spots.FRAME) + 1;


    %% Import tracks

    filtered_track_nodes = root.getElementsByTagName('FilteredTracks');
    if filtered_track_nodes.getLength < 1
        tracks = cell(0);
        return;
    end

    track_nodes = filtered_track_nodes.item(0).getElementsByTagName('TrackID');
    n_tracks = track_nodes.getLength;
    tracks = cell(n_tracks, 1);

    x_col = spots.FEATURES('POSITION_X');
    y_col = spots.FEATURES('POSITION_Y');
    z_col = spots.FEATURES('POSITION_Z');

    all_track_nodes = root.getElementsByTagName('AllTracks');
    track_item_nodes = all_track_nodes.item(0).getElementsByTagName('Track');
    track_map = containers.Map('KeyType', 'double', 'ValueType', 'any');
    for i = 0 : track_item_nodes.getLength - 1
        track_el = track_item_nodes.item(i);
        track_id = str2double(track_el.getAttribute('TRACK_ID'));
        edge_nodes = track_el.getElementsByTagName('Edge');
        n_edges = edge_nodes.getLength;
        
        spot_ids = NaN(n_edges + 1, 1);
        t_pos = NaN(n_edges + 1, 1);
        
        for j = 0 : n_edges - 1
            
            edge_el = edge_nodes.item(j);
            source_id = str2double(edge_el.getAttribute('SPOT_SOURCE_ID'));
            target_id = str2double(edge_el.getAttribute('SPOT_TARGET_ID'));
            
            if j == 0
                spot_ids(1) = source_id;
                t_pos(1) = spots.FRAME(source_id+1);
            end
            spot_ids(j+2) = target_id;
            t_pos(j+2) = spots.FRAME(target_id+1);
        end
        
        [ t_pos, S_index ] = sort(t_pos);
        spot_ids = spot_ids(S_index);
        
        xyz = NaN(numel(spot_ids), 3);
        
        xyz(:,1) = spots.all(spot_ids+1, x_col);
        xyz(:,2) = spots.all(spot_ids+1, y_col);
        xyz(:,3) = spots.all(spot_ids+1, z_col);
        
        track_map(track_id) = [ xyz t_pos * info.timeInverval ]; % store time in physical units
    end

    for i = 0 : n_tracks-1
        track_id = str2double(track_nodes.item(i).getAttribute('TRACK_ID'));
        tracks{i+1} = track_map(track_id);
    end

    if ~suppress_output
        fprintf('Done.\n');
    end

end