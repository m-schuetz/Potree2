
import {Potree, Mesh, Vector3, geometries} from "potree";
import {EventDispatcher, KeyCodes, MouseCodes} from "potree";

export class Measure{

	constructor(){
		
		this.markers = [];

	}

	addMarker(position){
		this.markers.push(position.clone());
	}

};

export class MeasureTool{

	constructor(potree){
		this.potree = potree;
		this.renderer = potree.renderer;
		this.element = potree.renderer.canvas;

		this.cursor = new Mesh("sphere", geometries.sphere);
		this.cursor.scale.set(10.5, 10.5, 10.5);
		this.cursor.renderLayer = 10;
		this.cursor.visible = false;
		potree.scene.root.children.push(this.cursor);

		this.currentMeasure = null;

		this.measures = [];

		potree.onUpdate(this.update.bind(this));

		this.dispatcher = new EventDispatcher();

	}

	reset(){

	}

	update(){
		let depth = camera.getWorldPosition().distanceTo(this.cursor.position);
		let radius = depth / 50;

		this.cursor.scale.set(radius, radius, radius);

		for(let measure of this.measures){

			for(let marker of measure.markers){

				let depth = camera.getWorldPosition().distanceTo(marker);
				let radius = depth / 50;

				this.renderer.drawSphere(marker, radius);
				// this.renderer.drawBox(
				// 	marker,
				// 	new Vector3(radius, radius, radius),
				// 	new Vector3(255, 255, 0),
				// );
			}

			for(let i = 0; i < measure.markers.length - 1; i++){
				this.renderer.drawLine(
					measure.markers[i + 0],
					measure.markers[i + 1],
					new Vector3(255, 0, 0),
				);
			}

		}
	}

	measureMove(e){
		let [x, y] = [e.event.clientX, e.event.clientY];

		let node = this.cursor;

		Potree.pick(x, y, (result) => {

			if(result.depth !== Infinity){
				node.position.copy(result.position);
				node.visible = true;

				Potree.pickPos = result.position;
			}
		});
	}

	onClick(e){
		console.log(e);
	}

	startMeasuring(args = {}){

		if(this.currentMeasure){
			this.stopMeasuring();
		}

		let maxMarkers = args.maxMarkers ?? Infinity;
		let requiredMarkers = args.requiredMarkers ?? null;

		let measure = new Measure();
		this.currentMeasurement = {measure, args};

		this.measures.push(measure);

		this.dispatcher.add("mousemove", (e) => {this.measureMove(e)});
		this.dispatcher.add("mouseup", (e) => {
			
			if(e.event.button === MouseCodes.LEFT && this.cursor.visible){

				let markerPos = this.cursor.position.clone();
			
				measure.addMarker(markerPos);

				if(measure.markers.length === maxMarkers){
					this.stopMeasuring();
				}else if(measure.markers.length === requiredMarkers){
					this.stopMeasuring();
				}

			}else if(e.event.button === MouseCodes.RIGHT){

				if(requiredMarkers && measure.markers.length !== requiredMarkers){
					this.measures.pop();
				}

				this.stopMeasuring();
			}

		});

	}

	stopMeasuring(){
		this.dispatcher.removeAll();
		this.cursor.visible = false;
	}

};
