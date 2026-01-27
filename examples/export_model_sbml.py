"""
Example: Exporting Synthetic Models to SBML Format

This example demonstrates how to create a synthetic virtual cell model
and export it to standard SBML (Systems Biology Markup Language) format.

SBML is a widely-used standard for representing biochemical networks,
allowing models to be shared between different simulation and analysis tools.
"""

from synthetic import Builder

# Create a virtual cell model
# [2, 5, 10] creates 2 cascades of degree 1, 5 of degree 2, 10 of degree 3
vc = Builder.specify([2, 5, 10], random_seed=42)

# Access the underlying ModelBuilder
model = vc.model

# Display model statistics
print("Model Statistics:")
print(model.head())

# Print some additional details
print(f"\nNumber of species: {len(model.get_state_variables())}")
print(f"Number of parameters: {len(model.get_parameters())}")
print(f"Number of reactions: {len(model.reactions)}")

# Export to SBML file
sbml_filename = 'synthetic_model.sbml'
model.save_sbml_model_as(sbml_filename)
print(f"\n✓ Model exported to SBML: {sbml_filename}")

# Export to Antimony format (human-readable)
antimony_filename = 'synthetic_model.ant'
model.save_antimony_model_as(antimony_filename)
print(f"✓ Model exported to Antimony: {antimony_filename}")

# Optionally get SBML as string for programmatic use
sbml_string = model.get_sbml_model()
print(f"\nSBML string length: {len(sbml_string)} characters")
print("First 200 characters of SBML:")
print(sbml_string[:200])

# Get Antimony as string for programmatic use
antimony_string = model.get_antimony_model()
print(f"\nAntimony string length: {len(antimony_string)} characters")

# Example: Save both formats with custom naming
print("\n" + "="*60)
print("Example with custom model name:")
print("="*60)

# Create another model with a specific name
vc2 = Builder.specify([3, 6, 12], name="MyCustomModel", random_seed=123)
model2 = vc2.model

# Export with descriptive filenames
model2.save_sbml_model_as('my_custom_model.sbml')
model2.save_antimony_model_as('my_custom_model.ant')

print("✓ Custom model exported to 'my_custom_model.sbml' and 'my_custom_model.ant'")

print("\nUsage Tips:")
print("- SBML files can be imported into COPASI, libRoadRunner, and other SBML-compliant tools")
print("- Antimony files are human-readable and easier to inspect/modify")
print("- Both formats preserve the complete model structure and parameters")
print("- Use SBML for interoperability, Antimony for readability")