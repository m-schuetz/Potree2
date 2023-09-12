import {EventDispatcher, SplatType} from "potree";
import {PointAttribute, PointAttributeTypes} from "potree";

const ListingSchemes = {
	LAS_CLASSIFICATION: {
		0:       { visible: true, name: 'never classified'  , color: [0.5,  0.5,  0.5,  1.0] },
		1:       { visible: true, name: 'unclassified'      , color: [0.5,  0.5,  0.5,  1.0] },
		2:       { visible: true, name: 'ground'            , color: [0.63, 0.32, 0.18, 1.0] },
		3:       { visible: true, name: 'low vegetation'    , color: [0.0,  1.0,  0.0,  1.0] },
		4:       { visible: true, name: 'medium vegetation' , color: [0.0,  0.8,  0.0,  1.0] },
		5:       { visible: true, name: 'high vegetation'   , color: [0.0,  0.6,  0.0,  1.0] },
		6:       { visible: true, name: 'building'          , color: [1.0,  0.66, 0.0,  1.0] },
		7:       { visible: true, name: 'low point(noise)'  , color: [1.0,  0.0,  1.0,  1.0] },
		8:       { visible: true, name: 'key-point'         , color: [1.0,  0.0,  0.0,  1.0] },
		9:       { visible: true, name: 'water'             , color: [0.0,  0.0,  1.0,  1.0] },
		12:      { visible: true, name: 'overlap'           , color: [1.0,  1.0,  0.0,  1.0] },
		DEFAULT: { visible: true, name: 'default'           , color: [0.3,  0.6,  0.6,  0.5] },
	},
	LAS_RETURN_NUMBER: {
		0:       { visible: true, name: '0'    , color: [0.5,  0.5,  0.5,  1.0] },
		1:       { visible: true, name: '1'    , color: [0.5,  0.5,  0.5,  1.0] },
		2:       { visible: true, name: '2'    , color: [0.63, 0.32, 0.18, 1.0] },
		3:       { visible: true, name: '3'    , color: [0.0,  1.0,  0.0,  1.0] },
		4:       { visible: true, name: '4'    , color: [0.0,  1.0,  0.0,  1.0] },
		5:       { visible: true, name: '5'    , color: [0.0,  1.0,  0.0,  1.0] },
		6:       { visible: true, name: '6'    , color: [0.0,  1.0,  0.0,  1.0] },
		7:       { visible: true, name: '7'    , color: [0.0,  1.0,  0.0,  1.0] },
		DEFAULT: { visible: true, name: 'default'           , color: [0.3,  0.6,  0.6,  0.5] },
	},
	NONE: {
		DEFAULT: { visible: true, name: 'default'           , color: [0.3,  0.6,  0.6,  0.5] },
	}
};

export class AttributeSettings{
	constructor(name){
		this.name = name;
		this.stats = null;
		this.runtime = true;
		this.mapping = null;
		this.range = null;
		this.filterRange = [-Infinity, Infinity];
		this.clamp = false;
		this.runtime = false;
		this.mapping = null;
		this.listing = ListingSchemes.LAS_CLASSIFICATION;
	}
};

export class PointCloudMaterial{

	constructor(){
		
		this.initialized = false;
		this.statsUpdated = false;
		this.attributes = new Map();
		this.attributeSettings = new Map();
		this.mappings = [];
		this.selectedMappings = new Map();

		this.needsCompilation = false;
		this.splatType = SplatType.POINTS;
		
		let dispatcher = new EventDispatcher();
		this.events = {
			dispatcher,
			onChange: (callback, args) => dispatcher.add("change", callback, args),
		};
	}

	recompile(){
		this.needsCompilation = true;
	}

	registerAttribute(attribute){

		if(this.attributes.has(attribute.name)){
			throw `an attribute with the id '${attribute.name}' is already registered`;
		}

		let settings = new AttributeSettings(attribute.name);

		this.attributes.set(attribute.name, attribute);
		this.attributeSettings.set(attribute.name, settings);

		this.recompile();
		this.events.dispatcher.dispatch("change", {material: this});
	}

	registerMapping(mapping){

		let index = 128 + this.mappings.length;
		// let mapping = {name, condition, inputs, wgsl, index};
		let {name, condition, inputs, wgsl} = mapping;
		mapping.index = index;

		this.mappings.push(mapping);

		this.recompile();

		for(let [name, attribute] of this.attributes){
			if(attribute && condition(attribute)){
				if(!this.selectedMappings.has(name)){
					this.selectedMappings.set(name, mapping);
				}
			}
		}
		
		this.events.dispatcher.dispatch("change", {material: this});

		return index;
	}

	update(pointcloud){

		let statsList = pointcloud?.root?.geometry?.statsList;

		if(statsList && !this.statsUpdated){

			for(let stats of statsList){

				if(this.attributes.has(stats.name)){
					let attribute = this.attributes.get(stats.name);

					attribute.stats = stats;

					// if(attribute instanceof Attribute_Scalar){
					// 	attribute.range = [stats.min, stats.max];
					// }
				}
			}

			{ // elevation

				let xyz = statsList.find(stats => ["XYZ", "position"].includes(stats.name));

				if(xyz){
					let stats = {
						name: "elevation",
						min: xyz.min[2],
						max: xyz.max[2],
					};
					let size = stats.max - stats.min;

					let attribute = this.attributes.get("elevation");
					attribute.stats = stats;
					attribute.clamp = true;
					attribute.range = [stats.min - 0.05 * size, stats.max + 0.05 * size];
				}
			}

			this.statsUpdated = true;

			this.events.dispatcher.dispatch("change", {material: this});
		}

		

	}

	init(pointcloud){

		// debugger;

		if(this.initialized){
			return;
		}

		for(let attribute of pointcloud.attributes.attributes){
			this.registerAttribute(attribute);
		}

		{ // elevation
			let attribute = new PointAttribute("elevation", PointAttributeTypes.UINT8, 1);
			attribute.byteOffset = 0;

			this.registerAttribute(attribute);
		}

		// console.log(statsList);
		console.log(this.attributes);

		this.initialized = true;

		this.events.dispatcher.dispatch("change", {material: this});
	}

};