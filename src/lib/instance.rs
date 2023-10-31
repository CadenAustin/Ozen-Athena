use anyhow::{anyhow, Result};
use log::*;
use std::collections::HashSet;
use winit::window::Window;

use vulkanalia::{
    prelude::v1_0::*, 
    vk::ExtDebugUtilsExtension, 
    window as vk_window
};

use crate::{
    app::{AppData, PORTABILITY_MACOS_VERSION, VALIDATION_ENABLED, VALIDATION_LAYER},
    debug::debug_callback,
};

pub(crate) unsafe fn create_instance(
    window: &Window,
    entry: &Entry,
    data: &mut AppData,
) -> Result<Instance> {
    let application_info = vk::ApplicationInfo::builder()
        .application_name(b"Vulkanalia Tutorial\0")
        .application_version(vk::make_version(1, 0, 0))
        .engine_name(b"No Engine\0")
        .engine_version(vk::make_version(1, 0, 0))
        .api_version(vk::make_version(1, 0, 0));

    let available_layers = entry
        .enumerate_instance_layer_properties()
        .unwrap()
        .iter()
        .map(|l| l.layer_name)
        .collect::<HashSet<_>>();

    if VALIDATION_ENABLED && !available_layers.contains(&VALIDATION_LAYER) {
        return Err(anyhow!("Validation Layer Not Supported"));
    }

    let layers = if VALIDATION_ENABLED {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        Vec::new()
    };

    let mut extensions = vk_window::get_required_instance_extensions(window)
        .iter()
        .map(|e| e.as_ptr())
        .collect::<Vec<_>>();

    if VALIDATION_ENABLED {
        extensions.push(vk::EXT_DEBUG_UTILS_EXTENSION.name.as_ptr());
    }

    //For MacOS vulkan instances
    let flags =
        if cfg!(target_os = "macos") && entry.version().unwrap() >= PORTABILITY_MACOS_VERSION {
            info!("Enabling extensions for MacOS portability.");
            extensions.push(
                vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_EXTENSION
                    .name
                    .as_ptr(),
            );
            extensions.push(vk::KHR_PORTABILITY_ENUMERATION_EXTENSION.name.as_ptr());
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        } else {
            vk::InstanceCreateFlags::empty()
        };

    let mut info = vk::InstanceCreateInfo::builder()
        .application_info(&application_info)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .flags(flags);

    let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
        .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
        .user_callback(Some(debug_callback));

    if VALIDATION_ENABLED {
        info = info.push_next(&mut debug_info);
    }

    let instance = entry.create_instance(&info, None).unwrap();

    if VALIDATION_ENABLED {
        data.messenger = instance
            .create_debug_utils_messenger_ext(&debug_info, None)
            .unwrap();
    }

    Ok(instance)
}
