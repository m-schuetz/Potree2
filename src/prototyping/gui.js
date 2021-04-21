
import * as dat from "dat.gui";

let gui = null;
let guiContent = {

	// INFOS
	"#points": "0",
	"#nodes": "0",
	"fps": "0",
	"duration(update)": "0",
	// "timings": "",
	"camera": "",


	// INPUT
	"show bounding box": false,
	"mode": "pixels",
	// "mode": "dilate",
	// "mode": "HQS",
	"attribute": "rgba",
	"point budget (M)": 1,
	"point size": 3,
	"update": true,

	// COLOR ADJUSTMENT
	"scalar min": 0,
	"scalar max": 2 ** 16,
	"gamma": 1,
	"brightness": 0,
	"contrast": 0,
};
window.guiContent = guiContent;
let guiAttributes = null;
let guiScalarMin = null;
let guiScalarMax = null;

export function initGUI(){

	gui = new dat.GUI();
	window.gui = gui;
	
	{
		let stats = gui.addFolder("stats");
		stats.open();
		stats.add(guiContent, "#points").listen();
		stats.add(guiContent, "#nodes").listen();
		stats.add(guiContent, "fps").listen();
		stats.add(guiContent, "duration(update)").listen();
		stats.add(guiContent, "camera").listen();
	}

	{
		let input = gui.addFolder("input");
		input.open();

		input.add(guiContent, "mode", [
			"pixels", 
			"dilate",
			"HQS",
		]);
		input.add(guiContent, "show bounding box");
		input.add(guiContent, "update");
		guiAttributes = input.add(guiContent, "attribute", ["rgba"]).listen();
		window.guiAttributes = guiAttributes;

		// slider
		input.add(guiContent, 'point budget (M)', 0.01, 5);
		input.add(guiContent, 'point size', 1, 5);
	}

	{
		let input = gui.addFolder("Color Adjustments");
		input.open();

		guiScalarMin = input.add(guiContent, 'scalar min', 0, 2 ** 16).listen();
		guiScalarMax = input.add(guiContent, 'scalar max', 0, 2 ** 16).listen();
		input.add(guiContent, 'gamma', 0, 2).listen();
		input.add(guiContent, 'brightness', -1, 1).listen();
		input.add(guiContent, 'contrast', -1, 1).listen();
	}

}