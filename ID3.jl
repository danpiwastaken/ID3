using DataFrames
using CSV
using Statistics
using Random

#df = DataFrame(CSV.File("iris.csv"))

function load_data(f_name::AbstractString)
    df = DataFrame(CSV.File(f_name))
    return df
end

function create_train_test(df::DataFrame,prop=2/3,randomseed=1234)
    df_training = similar(df,0)
    df_testing  = similar(df,0)

    # Now split the df into df_training and df_testing
    df_size = size(df,1)
    training_proportion = prop
    trainingsize = round(df_size*training_proportion)

    # Create a random permutation vector
    randvec = randperm!(MersenneTwister(randomseed),
                        Vector{Int64}(undef,df_size))

    for k in axes(df)[1]
        push!(  k ≤ trainingsize ?
                df_training : df_testing ,
                df[randvec[k],:]
        )
    end
    return (df_training,df_testing)
end

function element_freqs(array_of_elements)
    freqs = Dict()
    number_of_elements = length(array_of_elements)

    for (key, value) in counter(array_of_elements)
        freqs[key] = value / number_of_elements
    end

    return freqs
end

function compute_entropy(array_of_elements)
    freqs = element_freqs(array_of_elements)
    entropy = 0
    for (key, val) in freqs 
        entropy -= val * log2(val)
    end

    return entropy
end

function counter(array_of_elements)
    counts = Dict()

    for element in array_of_elements
        counts[element] = get(counts, element, 0) + 1
    end

    return counts
end

function highest_count(counts)
    max_val = 0
    max_count = ""

    for (key, val) in counts
        if val > max_val
            max_val = val
            max_count = key
        end
    end

    return max_count
end

function entropy_after_split(df::DataFrame, decision_feature, feature_to_split)
    feature_values = Set(df[!, feature_to_split])
    att_entropy = 0
    entropy_dict = Dict()

    for value in feature_values
        larger_filter = df[!, feature_to_split] .> value
        smaller_equal_filter = df[!, feature_to_split] .<= value
        
        S0_df = df[larger_filter, :]
        S1_df = df[smaller_equal_filter, :]
        
        H_S0 = compute_entropy(S0_df[!, decision_feature])
        H_S1 = compute_entropy(S1_df[!, decision_feature])

        H_S = (nrow(S0_df) / nrow(df)) * H_S0 + (nrow(S1_df) / nrow(df)) * H_S1

        entropy_dict[value] = H_S
    end

    att_entropy, index = findmin(collect(values(entropy_dict)))
    cutoff = collect(keys(entropy_dict))[index]
    return (att_entropy, cutoff)
end

function column_to_split(df::DataFrame, decision_feature)
    features = delete!(Set(names(df)), decision_feature)
    feature_to_split = nothing
    least_entropy = compute_entropy(df[!, decision_feature])
    cutoff = 0

    for feature in features
        split_entropy, curr_cutoff = entropy_after_split(df, decision_feature, feature)

        if split_entropy < least_entropy
            least_entropy = split_entropy
            feature_to_split = feature
            cutoff = curr_cutoff
        end
    end

    return feature_to_split, cutoff
end

function split_by(df::DataFrame, feature, cutoff)
    splits = Dict()

    larger_filter = df[!, feature] .> cutoff
    smaller_equal_filter = df[!, feature] .<= cutoff
        
    S0_df = df[larger_filter, :]
    splits[string(">", cutoff)] = S0_df

    S1_df = df[smaller_equal_filter, :]
    splits[string("<=", cutoff)] = S1_df

    return splits
end

function decision_tree(df::DataFrame, decision_feature)
    tree = Dict()
    split_column, cutoff = column_to_split(df, decision_feature)

    if split_column == nothing
        return highest_count(counter(df[!, decision_feature]))
    end

    tree[split_column] = Dict()
    for (key, val) in split_by(df, split_column, cutoff)
        if key == string("<=", cutoff)
            filter_condition = df[!, split_column] .<= cutoff
            filtered_df = df[filter_condition, :]
            tree[split_column][key] = decision_tree(filtered_df, decision_feature)
        elseif key == string(">", cutoff)
            filter_condition = df[!, split_column] .> cutoff
            filtered_df = df[filter_condition, :]
            tree[split_column][key] = decision_tree(filtered_df, decision_feature)
        end
    end

    return tree
end

function predict_decision(tree::AbstractDict, test_example)
    split = collect(keys(tree))[1]
    value = test_example[split]
    branch = nothing
    for key in collect(keys(tree[split]))
        cutoff = parse(Float64, key[end-2:end])
        if value <= cutoff
            branch = tree[split][string("<=", cutoff)]
        elseif value > cutoff
            branch = tree[split][straing(">", cutoff)]
        end
    end

    if !isa(branch, Dict)
        return branch
    end
    
    return predict_decision(branch, test_example)
end

function find_accuracy(tree::AbstractDict, test_example::DataFrame)
    correct, total = 0, 0
    for i in 1:nrow(test_example)
        total += 1
        if test_example[i, ncol(test)] == predict_decision(tree, test_example[i, :])
            correct += 1
        end
    end

    return correct/total
end

df = load_data("iris.csv")
train, test = create_train_test(df)
dtree = decision_tree(train, "Species")
println(find_accuracy(dtree, test))
