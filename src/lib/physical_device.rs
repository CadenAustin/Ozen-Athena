use std::collections::HashSet;

use log::{info, warn};

use thiserror::Error;

use anyhow::{anyhow, Result};

use vulkanalia::{
    prelude::v1_0::*,
    Instance,
    vk::KhrSurfaceExtension
};

use crate::{
    app::{AppData, DEVICE_EXTENSIONS},
    swapchain::SwapchainSupport,
    msaa::get_max_msaa_samples
};

#[derive(Debug, Error)]
#[error("Missing {0}.")]
pub(crate)  struct SutibilityError(pub(crate) &'static str);

pub(crate) unsafe fn check_physical_device(
    instance: &Instance,
    data: &AppData,
    physical_device: vk::PhysicalDevice,
) -> Result<()> {
    QueueFamilyIndices::get(instance, data, physical_device).unwrap();

    check_physical_device_extensions(instance, physical_device).unwrap();

    let support = SwapchainSupport::get(instance, data, physical_device).unwrap();
    if support.formats.is_empty() || support.present_modes.is_empty() {
        return Err(anyhow!(SutibilityError("Insufficient Swapchain Support")));
    }

    let features = instance.get_physical_device_features(physical_device);
    if features.sampler_anisotropy != vk::TRUE {
        return Err(anyhow!(SutibilityError("No sampler anisotropy.")));
    }

    Ok(())
}

pub(crate) unsafe fn check_physical_device_extensions(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<()> {
    let extensions = instance
        .enumerate_device_extension_properties(physical_device, None)
        .unwrap()
        .iter()
        .map(|e| e.extension_name)
        .collect::<HashSet<_>>();

    if DEVICE_EXTENSIONS.iter().all(|e| extensions.contains(e)) {
        Ok(())
    } else {
        Err(anyhow!(SutibilityError("Missing Device Extensions")))
    }
}

pub(crate) unsafe fn pick_physical_device(instance: &Instance, data: &mut AppData) -> Result<()> {
    for physical_device in instance.enumerate_physical_devices().unwrap() {
        let properties = instance.get_physical_device_properties(physical_device);

        if let Err(error) = check_physical_device(instance, data, physical_device) {
            warn!(
                "Skipping Physical Device (`{}`): {}",
                properties.device_name, error
            );
        } else {
            info!("Selected Physical Device (`{}`)", properties.device_name);
            data.physical_device = physical_device;
            data.msaa_samples = get_max_msaa_samples(instance, data);
            return Ok(());
        }
    }
    Err(anyhow!("Failed to Find Physical Device"))
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct QueueFamilyIndices {
    pub(crate) graphics: u32,
    pub(crate) present: u32,
}

impl QueueFamilyIndices {
    pub(crate) unsafe fn get(
        instance: &Instance,
        data: &AppData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        let properties = instance.get_physical_device_queue_family_properties(physical_device);

        let graphics = properties
            .iter()
            .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|i| i as u32);

        let mut present = None;
        for (index, _) in properties.iter().enumerate() {
            if instance
                .get_physical_device_surface_support_khr(
                    physical_device,
                    index as u32,
                    data.surface,
                )
                .unwrap()
            {
                present = Some(index as u32);
                break;
            }
        }

        if let (Some(graphics), Some(present)) = (graphics, present) {
            Ok(Self { graphics, present })
        } else {
            Err(anyhow!(SutibilityError("Missing Queue Family: Graphics")))
        }
    }
}
