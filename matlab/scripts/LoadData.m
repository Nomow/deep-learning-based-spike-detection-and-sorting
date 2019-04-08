% loads data from directory
function [data] = LoadData(directory_path, file_name)
    % loads all data from directory
    if(nargin == 1)
        files =  dir(directory_path);
        file_name = {files.name}';
        file_name(1:2) = [];
        vertices = cell(size(file_name, 1), 1);
        faces = cell(size(file_name, 1), 1);
        parfor i = 1:size(file_name,1)
            full_path = fullfile(directory_path, file_name{i});
            data{i} = load(full_path)
        end
    % loads single data
    else
        full_path = fullfile(directory_path, file_name);
        data{i} = load(full_path)
    end
end