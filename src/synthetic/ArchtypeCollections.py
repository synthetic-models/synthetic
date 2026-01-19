from .ReactionArchtype import ReactionArchtype

michaelis_menten = ReactionArchtype(
    'Michaelis Menten',
    ('&S',), ('&E',),
    ('Km', 'Vmax'),
    'Vmax*&S/(Km + &S)',
    assume_parameters_values={'Km': 100, 'Vmax': 10},
    assume_reactant_values={'&S': 100},
    assume_product_values={'&E': 0})

michaelis_menten_fixed = ReactionArchtype(
    'Michaelis Menten',
    ('&S',), ('&E',),
    (),
    '100*&S/(1 + &S)',
    assume_reactant_values={'&S': 100},
    assume_product_values={'&E': 0})

mass_action_21 = ReactionArchtype(
    'Mass Action',
    ('&A', '&B'), ('&C',),
    ('ka', 'kd'),
    'ka*&A*&B',
    assume_parameters_values={'ka': 0.001, 'kd': 0.01},
    assume_reactant_values={'&A': 100, '&B': 100},
    assume_product_values={'&C': 0},
    reversible=True,
    reverse_rate_law='kd*&C')

simple_rate_law = ReactionArchtype(
    'Simple Rate Law',
    ('&A',), ('&B',),
    ('kf',),
    'kf*&A',
    assume_parameters_values={'kf': 0.01},
    assume_reactant_values={'&A': 100},
    assume_product_values={'&B': 0})

michaelis_menten_stim = ReactionArchtype(
    'Michaelis Menten',
    ('&S',), ('&E',),
    ('Km', 'Vmax'),
    'Vmax*&S*&I/(Km + &S)',
    extra_states=('&I',),
    assume_parameters_values={'Km': 100, 'Vmax': 10},
    assume_reactant_values={'&S': 100},
    assume_product_values={'&E': 0})

michaelis_menten_inh_allosteric = ReactionArchtype(
    'Michaelis Menten',
    ('&S',), ('&E',),
    ('Km', 'Vmax', 'Ki'),
    'Vmax*&S/(Km + &S)*(1+&I*Ki)',
    extra_states=('&I',),
    assume_parameters_values={'Km': 100, 'Vmax': 10, 'Ki': 0.1},
    assume_reactant_values={'&S': 100},
    assume_product_values={'&E': 0})

michaelis_menten_inh_competitive_1 = ReactionArchtype(
    'Michaelis Menten',
    ('&S',), ('&E',),
    ('Km', 'Vmax', 'Ki'),
    'Vmax*&S/(Km*(1+&I*Ki) + &S)',
    extra_states=('&I',),
    assume_parameters_values={'Km': 100, 'Vmax': 10, 'Ki': 0.1},
    assume_reactant_values={'&S': 100},
    assume_product_values={'&E': 0})

degredation = ReactionArchtype(
    'Degredation',
    ('&A',), (),
    ('Kdeg',),
    'Kdeg*&A',
    assume_parameters_values={'Kdeg': 0.01},
    assume_reactant_values={'&A': 100},
    assume_product_values={})

synthesis = ReactionArchtype(
    'Synthesis',
    (), ('&A',),
    ('Ksyn',),
    'Ksyn',
    assume_parameters_values={'Ksyn': 0.1},
    assume_reactant_values={},
    assume_product_values={'&A': 0})

def create_archtype_synthesis(allo_stimulators=0, allo_inhibitors=0):
    reactants = ()
    products = ('&R',)
    parameters = ('Ksyn',)
    rate_law = 'Ksyn'
    extra_states = ()
    assume_reactant_values = {}
    assume_product_values = {'&R': 0}
    assume_parameters_values = {'Ksyn': 0.1}
    total_extra_states = ()
    if allo_stimulators > 0:
        # add the stimulators to the equation
        stim_str = '*('
        for i in range(allo_stimulators):
            stim_str += f'&A{i}*Ks{i}+'
        
        rate_law += stim_str[:-1] + ')'

        # fill extra states
        extra_states = tuple([f'&A{i}' for i in range(allo_stimulators)])
        total_extra_states += extra_states

        # fill parameters
        parameters += tuple([f'Ks{i}' for i in range(allo_stimulators)])

        # fill assume parameters values
        assume_parameters_values.update({f'Ks{i}': 1e-4 for i in range(allo_stimulators)})

    if allo_inhibitors > 0:
        # add the inhibitors to the equation
        inh_str = '*(1/(1+'
        for i in range(allo_inhibitors):
            inh_str += f'&I{i}*Ki{i}+'
        
        rate_law += inh_str[:-1] + '))'

        # fill extra states
        extra_states += tuple([f'&I{i}' for i in range(allo_inhibitors)])
        total_extra_states += extra_states

        # fill parameters
        parameters += tuple([f'Ki{i}' for i in range(allo_inhibitors)])

        # fill assume parameters values
        assume_parameters_values.update({f'Ki{i}': 1e-4 for i in range(allo_inhibitors)})

    return ReactionArchtype(
        'Synthesis',
        reactants, products,
        parameters,
        rate_law,
        extra_states=extra_states,
        assume_reactant_values=assume_reactant_values,
        assume_product_values=assume_product_values,
        assume_parameters_values=assume_parameters_values)

def create_archtype_degredation(allo_stimulators=0, allo_inhibitors=0):
    """
    Creates a degredation reaction archtype with the given number of
    stimulators and inhibitors.
    """
    reactants = ('&R',)
    products = ()
    parameters = ('Kdeg',)
    rate_law = 'Kdeg*&R'
    assume_reactant_values = {'&R': 100}
    assume_product_values = {}
    extra_states = ()
    assume_parameters_values = {'Kdeg': 0.01}
    total_extra_states = ()

    if allo_stimulators > 0:
        # add the stimulators to the equation
        stim_str = '*('
        for i in range(allo_stimulators):
            stim_str += f'&A{i}*Ks{i}+'
        
        rate_law += stim_str[:-1] + ')'

        # fill extra states
        extra_states = tuple([f'&A{i}' for i in range(allo_stimulators)])
        total_extra_states += extra_states

        # fill parameters
        parameters += tuple([f'Ks{i}' for i in range(allo_stimulators)])

        # fill assume parameters values
        assume_parameters_values.update({f'Ks{i}': 1e-4 for i in range(allo_stimulators)})

    if allo_inhibitors > 0:
        # add the inhibitors to the equation
        inh_str = '*(1/(1+'
        for i in range(allo_inhibitors):
            inh_str += f'&I{i}*Ki{i}+'
        
        rate_law += inh_str[:-1] + '))'

        # fill extra states
        extra_states += tuple([f'&I{i}' for i in range(allo_inhibitors)])
        total_extra_states += extra_states

        # fill parameters
        parameters += tuple([f'Ki{i}' for i in range(allo_inhibitors)])

        # fill assume parameters values
        assume_parameters_values.update({f'Ki{i}': 1e-4 for i in range(allo_inhibitors)})


    return ReactionArchtype(
        'Degredation General',
        reactants, products,
        parameters,
        rate_law,
        extra_states=extra_states,
        assume_parameters_values=assume_parameters_values,
        assume_reactant_values=assume_reactant_values,
        assume_product_values=assume_product_values)

def create_archtype_mass_action(
                    reactant_count=1, 
                    product_count=1, 
                    allo_stimulators=0, 
                    additive_stimulators=0, 
                    allo_inhibitors=0, 
                    comp_inhibitors=0,
                    rev_allo_stimulators=0,
                    rev_additive_stimulators=0,
                    rev_allo_inhibitors=0,
                    rev_comp_inhibitors=0,):
    reactants = tuple(f'&R{i}' for i in range(reactant_count))
    products = tuple(f'&P{i}' for i in range(product_count))
    parameters = ('Ka', 'Kd')
    assume_parameters_values={'Ka': 0.001, 'Kd': 0.01}
    forw_rate_law = 'Ka'
    for i in range(reactant_count):
        forw_rate_law += f'*&R{i}'
    rev_rate_law = 'Kd'
    for i in range(product_count):
        rev_rate_law += f'*&P{i}'

    total_extra_states = ()

    if allo_stimulators > 0:
        # add the stimulators to the equation
        stim_str = '*('
        for i in range(allo_stimulators):
            stim_str += f'&A{i}*Ks{i}+'
        
        forw_rate_law += stim_str[:-1] + ')'

        # fill extra states
        extra_states = tuple([f'&A{i}' for i in range(allo_stimulators)])
        total_extra_states += extra_states

        # fill parameters
        parameters += tuple([f'Ks{i}' for i in range(allo_stimulators)])

        # fill assume parameters values
        assume_parameters_values.update({f'Ks{i}': 1e-4 for i in range(allo_stimulators)})

    if additive_stimulators > 0:
        # weak stimulators represent that stimulant is not required for the reaction to occur
        stim_weak_str = '(Ka+'
        for i in range(additive_stimulators):
            stim_weak_str += f'&W{i}*Kw{i}+'
        
        forw_rate_law = stim_weak_str[:-1] + ')' + forw_rate_law[2:]

        # fill extra states
        extra_states = tuple([f'&W{i}' for i in range(additive_stimulators)])
        total_extra_states += extra_states

        # fill parameters
        parameters += tuple([f'Kw{i}' for i in range(additive_stimulators)])

        # fill assume parameters values
        assume_parameters_values.update({f'Kw{i}': 1e-4 for i in range(additive_stimulators)})

    if allo_inhibitors > 0:
        # add the inhibitors to the equation, exact same thing as allo_stimulators but 
        # applied to the reverse rate law, with different parameter names 

        # add the stimulators to the equation
        altered_str = '*(1/(1+'
        for i in range(allo_inhibitors):
            altered_str += f'&I{i}*Ki{i}+'

        forw_rate_law += altered_str[:-1] + '))'

        # fill extra states
        extra_states = tuple([f'&I{i}' for i in range(allo_inhibitors)])
        total_extra_states += extra_states

        # fill parameters
        parameters += tuple([f'Ki{i}' for i in range(allo_inhibitors)])

        # fill assume parameters values
        assume_parameters_values.update({f'Ki{i}': 1e-4 for i in range(allo_inhibitors)})

    if comp_inhibitors > 0:
        # add the inhibitors to the equation, exact same thing as addi_stimulators but
        # applied to the reverse rate law, with different parameter names

        assert additive_stimulators == 0, 'Currently this function does not support having both additive and competitive inhibitors'

        # add the stimulators to the equation
        stim_str = '(Ka/(1+'
        for i in range(comp_inhibitors):
            stim_str += f'&C{i}*Kc{i}+'

        forw_rate_law = stim_str[:-1] + '))' + forw_rate_law[2:]

        # fill extra states
        extra_states = tuple([f'&C{i}' for i in range(comp_inhibitors)])
        total_extra_states += extra_states

        # fill parameters
        parameters += tuple([f'Kc{i}' for i in range(comp_inhibitors)])

        # fill assume parameters values
        assume_parameters_values.update({f'Kc{i}': 1e-4 for i in range(comp_inhibitors)})

    # reverse rate law MODIFICATIONS

    if rev_allo_stimulators > 0:
        # add the stimulators to the equation
        stim_str = '*('
        for i in range(rev_allo_stimulators):
            stim_str += f'?A{i}*Ksr{i}+'
        
        rev_rate_law += stim_str[:-1] + ')'

        # fill extra states
        extra_states = tuple([f'?A{i}' for i in range(rev_allo_stimulators)])
        total_extra_states += extra_states

        # fill parameters
        parameters += tuple([f'Ksr{i}' for i in range(rev_allo_stimulators)])

        # fill assume parameters values
        assume_parameters_values.update({f'rKs{i}': 1e-4 for i in range(rev_allo_stimulators)})

    if rev_additive_stimulators > 0:
        # weak stimulators represent that stimulant is not required for the reaction to occur
        stim_weak_str = '(Kd+'
        for i in range(rev_additive_stimulators):
            stim_weak_str += f'?W{i}*Kwr{i}+'
        
        rev_rate_law = stim_weak_str[:-1] + ')' + rev_rate_law[2:]

        # fill extra states
        extra_states = tuple([f'?W{i}' for i in range(rev_additive_stimulators)])
        total_extra_states += extra_states

        # fill parameters
        parameters += tuple([f'Kwr{i}' for i in range(rev_additive_stimulators)])

        # fill assume parameters values
        assume_parameters_values.update({f'Kwr{i}': 1e-4 for i in range(rev_additive_stimulators)})

    if rev_allo_inhibitors > 0:
        # add the inhibitors to the equation, exact same thing as allo_stimulators but 
        # applied to the reverse rate law, with different parameter names 

        # add the stimulators to the equation
        altered_str = '*(1/(1+'
        for i in range(rev_allo_inhibitors):
            altered_str += f'?I{i}*Kir{i}+'

        rev_rate_law += altered_str[:-1] + '))'

        # fill extra states
        extra_states = tuple([f'?I{i}' for i in range(rev_allo_inhibitors)])
        total_extra_states += extra_states

        # fill parameters
        parameters += tuple([f'Kir{i}' for i in range(rev_allo_inhibitors)])

        # fill assume parameters values
        assume_parameters_values.update({f'Kir{i}': 1e-4 for i in range(rev_allo_inhibitors)})

    if rev_comp_inhibitors > 0:
        # add the inhibitors to the equation, exact same thing as addi_stimulators but
        # applied to the reverse rate law, with different parameter names

        assert rev_additive_stimulators == 0, 'Currently this function does not support having both additive and competitive inhibitors'

        # add the stimulators to the equation
        stim_str = '(Kd/(1+'
        for i in range(rev_comp_inhibitors):
            stim_str += f'?C{i}*Kcr{i}+'

        rev_rate_law = stim_str[:-1] + '))' + rev_rate_law[2:]

        # fill extra states
        extra_states = tuple([f'?C{i}' for i in range(rev_comp_inhibitors)])
        total_extra_states += extra_states

        # fill parameters
        parameters += tuple([f'Kcr{i}' for i in range(rev_comp_inhibitors)])

        # fill assume parameters values
        assume_parameters_values.update({f'Kcr{i}': 1e-4 for i in range(rev_comp_inhibitors)})

    return ReactionArchtype(
        'Mass Action General',
        reactants, products,
        parameters,
        forw_rate_law,
        extra_states=total_extra_states,
        assume_parameters_values=assume_parameters_values,
        assume_reactant_values={f'&R{i}': 100 for i in range(reactant_count)},
        assume_product_values={f'&P{i}': 0 for i in range(product_count)},
        reversible=True,
        reverse_rate_law=rev_rate_law)


def create_archtype_basal_michaelis(stimulators=0, stimulator_weak=0, allosteric_inhibitors=0, competitive_inhibitors=0):
    '''
    WARNING: basal currently only supports stimulator_weak and competitive_inhibitors
    The basal version do not generate Vmax as a parameter and only uses stimuator_weak and competitive_inhibitors
    '''
    if stimulators + allosteric_inhibitors + competitive_inhibitors + stimulator_weak == 0:
        return michaelis_menten

    # create the archtype

    archtype_name = 'Michaelis Menten General'

    reactants = ('&S',)
    products = ('&E',)
    upper_equation = 'Kc*&S'
    lower_equation = '(Km + &S)'
    total_extra_states = ()
    parameters = ('Km','Kc')
    assume_parameters_values={'Km': 100, 'Kc': 1}

    if stimulators > 0:
        # add the stimulators to the equation
        
        # first remove Kc from parameters
        parameters = tuple(p for p in parameters if p != "Kc")
        assume_parameters_values.pop("Kc", None)

        stim_str = '*('
        for i in range(stimulators):
            stim_str += f'&A{i}*Ka{i}+'
        
        upper_equation += stim_str[:-1] + ')'

        # fill extra states
        extra_states = tuple([f'&A{i}' for i in range(stimulators)])
        total_extra_states += extra_states

        # fill parameters
        parameters += tuple([f'Ka{i}' for i in range(stimulators)])

        # fill assume parameters values
        assume_parameters_values.update({f'Ka{i}': 0.01 for i in range(stimulators)})

    if stimulator_weak > 0:
        # weak stimulators represent that stimulant is not required for the reaction to occur
        
        stim_weak_str = '(Kc+'
        for i in range(stimulator_weak):
            stim_weak_str += f"Kc{i}*&W{i}+"
        
        upper_equation = stim_weak_str[:-1] + ")" + "*&S"

        # fill extra states
        extra_states = tuple([f'&W{i}' for i in range(stimulator_weak)])
        total_extra_states += extra_states

        # fill parameters
        parameters += tuple([f'Kc{i}' for i in range(stimulator_weak)])

        # fill assume parameters values
        assume_parameters_values.update({f'Kc{i}': 0.1 for i in range(stimulator_weak)})

    if allosteric_inhibitors > 0:
        # add the allosteric inhibitors to the equation
        inhb_allo_str = '/(1 + '
        for i in range(allosteric_inhibitors):
            inhb_allo_str += f'&L{i}/Ki{i} + '
        
        inhb_allo_str = inhb_allo_str[:-3] + ')'

        lower_equation += inhb_allo_str

        # fill extra states
        extra_states = tuple([f'&L{i}' for i in range(allosteric_inhibitors)])
        total_extra_states += extra_states

        # fill parameters
        parameters += tuple([f'Ki{i}' for i in range(allosteric_inhibitors)])

        # fill assume parameters values
        assume_parameters_values.update({f'Ki{i}': 0.01 for i in range(allosteric_inhibitors)})
    
    if competitive_inhibitors > 0:
        # add the competitive inhibitors to the equation
        inhb_comp_str = '(Km*(1+'
        for i in range(competitive_inhibitors):
            inhb_comp_str += f'&I{i}*Kic{i}+'
        
        inhb_comp_str = inhb_comp_str[:-1] + ')'

        lower_equation = inhb_comp_str + lower_equation[3:]

        # fill extra states
        extra_states = tuple([f'&I{i}' for i in range(competitive_inhibitors)])
        total_extra_states += extra_states

        # fill parameters
        parameters += tuple([f'Kic{i}' for i in range(competitive_inhibitors)])

        # fill assume parameters values
        assume_parameters_values.update({f'Kic{i}': 0.1 for i in range(competitive_inhibitors)})

    full_equation = f'{upper_equation}/{lower_equation}'

    general_reaction = ReactionArchtype(
        archtype_name,
        reactants, products,
        parameters,
        full_equation,
        extra_states=total_extra_states,
        assume_parameters_values=assume_parameters_values,
        assume_reactant_values={'&S': 100},
        assume_product_values={'&E': 0})
    
    return general_reaction

def create_archtype_michaelis_menten_v2(stimulators=0, stimulator_weak=0, allosteric_inhibitors=0, competitive_inhibitors=0): 
    '''
    WARNING: v2 currently only supports stimulator_weak and competitive_inhibitors
    The v2 version do not generate Vmax as a parameter and only uses stimuator_weak and competitive_inhibitors
    '''
    if stimulators + allosteric_inhibitors + competitive_inhibitors + stimulator_weak == 0:
        return michaelis_menten

    # create the archtype

    archtype_name = 'Michaelis Menten General'

    reactants = ('&S',)
    products = ('&E',)
    upper_equation = 'Kc*&S'
    lower_equation = '(Km + &S)'
    total_extra_states = ()
    parameters = ('Km','Kc')
    assume_parameters_values={'Km': 100, 'Kc': 1}

    if stimulators > 0:
        # add the stimulators to the equation
        
        # first remove Kc from parameters
        parameters = tuple(p for p in parameters if p != "Kc")
        assume_parameters_values.pop("Kc", None)

        stim_str = '*('
        for i in range(stimulators):
            stim_str += f'&A{i}*Ka{i}+'
        
        upper_equation += stim_str[:-1] + ')'

        # fill extra states
        extra_states = tuple([f'&A{i}' for i in range(stimulators)])
        total_extra_states += extra_states

        # fill parameters
        parameters += tuple([f'Ka{i}' for i in range(stimulators)])

        # fill assume parameters values
        assume_parameters_values.update({f'Ka{i}': 0.01 for i in range(stimulators)})

    if stimulator_weak > 0:
        # weak stimulators represent that stimulant is not required for the reaction to occur
        
        # first remove Kc from parameters
        parameters = tuple(p for p in parameters if p != 'Kc')
        assume_parameters_values.pop('Kc', None)
        
        stim_weak_str = '('
        for i in range(stimulator_weak):
            stim_weak_str += f"Kc{i}*&W{i}+"
        
        upper_equation = stim_weak_str[:-1] + ")" + "*&S"

        # fill extra states
        extra_states = tuple([f'&W{i}' for i in range(stimulator_weak)])
        total_extra_states += extra_states

        # fill parameters
        parameters += tuple([f'Kc{i}' for i in range(stimulator_weak)])

        # fill assume parameters values
        assume_parameters_values.update({f'Kc{i}': 0.1 for i in range(stimulator_weak)})

    if allosteric_inhibitors > 0:
        # add the allosteric inhibitors to the equation
        inhb_allo_str = '/(1 + '
        for i in range(allosteric_inhibitors):
            inhb_allo_str += f'&L{i}/Ki{i} + '
        
        inhb_allo_str = inhb_allo_str[:-3] + ')'

        lower_equation += inhb_allo_str

        # fill extra states
        extra_states = tuple([f'&L{i}' for i in range(allosteric_inhibitors)])
        total_extra_states += extra_states

        # fill parameters
        parameters += tuple([f'Ki{i}' for i in range(allosteric_inhibitors)])

        # fill assume parameters values
        assume_parameters_values.update({f'Ki{i}': 0.01 for i in range(allosteric_inhibitors)})
    
    if competitive_inhibitors > 0:
        # add the competitive inhibitors to the equation
        inhb_comp_str = '(Km*(1+'
        for i in range(competitive_inhibitors):
            inhb_comp_str += f'&I{i}*Kic{i}+'
        
        inhb_comp_str = inhb_comp_str[:-1] + ')'

        lower_equation = inhb_comp_str + lower_equation[3:]

        # fill extra states
        extra_states = tuple([f'&I{i}' for i in range(competitive_inhibitors)])
        total_extra_states += extra_states

        # fill parameters
        parameters += tuple([f'Kic{i}' for i in range(competitive_inhibitors)])

        # fill assume parameters values
        assume_parameters_values.update({f'Kic{i}': 0.1 for i in range(competitive_inhibitors)})

    full_equation = f'{upper_equation}/{lower_equation}'

    general_reaction = ReactionArchtype(
        archtype_name,
        reactants, products,
        parameters,
        full_equation,
        extra_states=total_extra_states,
        assume_parameters_values=assume_parameters_values,
        assume_reactant_values={'&S': 100},
        assume_product_values={'&E': 0})
    
    return general_reaction


def create_archtype_michaelis_menten(stimulators=0, stimulator_weak=0, allosteric_inhibitors=0, competitive_inhibitors=0):

    if stimulators + allosteric_inhibitors + competitive_inhibitors + stimulator_weak == 0:
        return michaelis_menten

    # create the archtype

    archtype_name = 'Michaelis Menten General'

    reactants = ('&S',)
    products = ('&E',)
    upper_equation = 'Vmax*&S'
    lower_equation = '(Km + &S)'
    total_extra_states = ()
    parameters = ('Km', 'Vmax')
    assume_parameters_values={'Km': 100, 'Vmax': 1}

    if stimulators > 0:
        # add the stimulators to the equation
        stim_str = '*('
        for i in range(stimulators):
            stim_str += f'&A{i}*Ka{i}+'
        
        upper_equation += stim_str[:-1] + ')'

        # fill extra states
        extra_states = tuple([f'&A{i}' for i in range(stimulators)])
        total_extra_states += extra_states

        # fill parameters
        parameters += tuple([f'Ka{i}' for i in range(stimulators)])

        # fill assume parameters values
        assume_parameters_values.update({f'Ka{i}': 0.01 for i in range(stimulators)})

    if stimulator_weak > 0:
        # weak stimulators represent that stimulant is not required for the reaction to occur
        stim_weak_str = '(Vmax+'
        for i in range(stimulator_weak):
            stim_weak_str += f'&W{i}*Kw{i}+'
        
        upper_equation = stim_weak_str[:-1] + ')' + upper_equation[4:]

        # fill extra states
        extra_states = tuple([f'&W{i}' for i in range(stimulator_weak)])
        total_extra_states += extra_states

        # fill parameters
        parameters += tuple([f'Kw{i}' for i in range(stimulator_weak)])

        # fill assume parameters values
        assume_parameters_values.update({f'Kw{i}': 0.1 for i in range(stimulator_weak)})

    if allosteric_inhibitors > 0:
        # add the allosteric inhibitors to the equation
        inhb_allo_str = '*(1+'
        for i in range(allosteric_inhibitors):
            inhb_allo_str += f'&L{i}*Kil{i}+'
        
        inhb_allo_str = inhb_allo_str[:-1] + ')'

        lower_equation += inhb_allo_str

        # fill extra states
        extra_states = tuple([f'&L{i}' for i in range(allosteric_inhibitors)])
        total_extra_states += extra_states

        # fill parameters
        parameters += tuple([f'Kil{i}' for i in range(allosteric_inhibitors)])

        # fill assume parameters values
        assume_parameters_values.update({f'Kil{i}': 0.01 for i in range(allosteric_inhibitors)})
    
    if competitive_inhibitors > 0:
        # add the competitive inhibitors to the equation
        inhb_comp_str = '(Km*(1+'
        for i in range(competitive_inhibitors):
            inhb_comp_str += f'&I{i}*Kic{i}+'
        
        inhb_comp_str = inhb_comp_str[:-1] + ')'

        lower_equation = inhb_comp_str + lower_equation[3:]

        # fill extra states
        extra_states = tuple([f'&I{i}' for i in range(competitive_inhibitors)])
        total_extra_states += extra_states

        # fill parameters
        parameters += tuple([f'Kic{i}' for i in range(competitive_inhibitors)])

        # fill assume parameters values
        assume_parameters_values.update({f'Kic{i}': 0.1 for i in range(competitive_inhibitors)})

    full_equation = f'{upper_equation}/{lower_equation}'

    general_reaction = ReactionArchtype(
        archtype_name,
        reactants, products,
        parameters,
        full_equation,
        extra_states=total_extra_states,
        assume_parameters_values=assume_parameters_values,
        assume_reactant_values={'&S': 100},
        assume_product_values={'&E': 0})

    return general_reaction
