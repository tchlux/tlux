on run argv
    -- Get the URL passed in from the command line.
    set theURL to item 1 of argv
    -- Open and load the page with Safari, get its contents, close the tab, and return the contents.
    tell application "Safari"
        -- Open the URL provided.
        open location theURL
        -- Make sure the source has loaded.
        set theHTML to the source of the front document
        set prevHTML to the source of the front document
        repeat while (the length of theHTML is 0) or (theHTML does not equal prevHTML)
            delay 0.1
            set prevHTML to theHTML
            set theHTML to the source of the front document
            -- Check if the loaded content is an image.
            set contentType to do JavaScript "document.contentType" in the front document
            if contentType starts with "image/" then
                -- If it's an image, return the URL (as the HTML source is not applicable).
                close the current tab of the front window
                return theURL
            end if
        end repeat
        -- Make sure all JavaScript has loaded if it exists.
        repeat while (do JavaScript "document.readyState" in the front document) is not "complete"
            delay 0.1
            set theHTML to the source of the front document
        end repeat
        -- Close the page that was opened.
        close the current tab of the front window
        -- Return the source contents of the URL.
        return theHTML
    end tell
end run
