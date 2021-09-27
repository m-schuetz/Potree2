
import * as dat from "dat.gui";
import {Potree} from "potree";

let gui = null;
let guiContent = {

	// INFOS
	"duration(update)": "0",
	"cam.pos": "",
	"cam.target": "",
	"cam.dir": "",

	// INPUT
	"show bounding box": false,
	"dilate": false,
	"Eye-Dome-Lighting": false,
	"point size": 3,
	"update": true,
	"resolution": 1_000,
	"primitive": "triangles",
};

// window.guiContent = guiContent;
let guiAttributes = null;
let guiScalarMin = null;
let guiScalarMax = null;

export function initGUI(potree){

	gui = new dat.GUI();
	window.gui = gui;
	window.guiContent = guiContent;
	
	{
		let stats = gui.addFolder("stats");
		stats.open();
		stats.add(guiContent, "cam.pos").listen();
		stats.add(guiContent, "cam.target").listen();
		stats.add(guiContent, "cam.dir").listen();
	}

	{
		let input = gui.addFolder("input");
		input.open();

		input.add(guiContent, "resolution", 10, 2_000).listen();
		input.add(guiContent, "primitive", ["triangles", "points"]).listen();
	}


	potree.addEventListener("update", () => {

		let state = Potree.state;

		guiContent["cam.pos"]   = state.camPos;
		guiContent["cam.target"]   = state.camTarget;
		guiContent["cam.dir"]   = state.camDir;

		Potree.settings.resolution = guiContent["resolution"];
		Potree.settings.primitive = guiContent["primitive"];

	
	});

}