#!/bin/bash

# List of prompts
prompts=(
  "Woman exercising by skipping a rope"
  "This is the best recipe I have ever tried for Cuban bread. I lived in Key West... Cuban Recipes, Bread Recipes, Cooking Recipes, Cuban Desserts, Pan Cubano Recipe, Cuban Bread, Cuban Sandwich, Sandwiches, Recipe From Scratch"
  "Photography lenses are on the table."
  "Shoe's the Word Art Print"
  "Red Square is a city square in Moscow, Russia."
  "7 personal finance tips every small business owner needs"
)

# Create output directory
mkdir -p outputs_infer

# Loop through each prompt and run inference
for i in "${!prompts[@]}"; do
  prompt="${prompts[$i]}"
  # Generate a safe filename
  filename="output_$((i+1)).png"
  echo "Running inference for: \"$prompt\""
  python infer.py --prompt "$prompt" --output "outputs_infer/$filename"
done

echo "All outputs saved in ./outputs_infer/"
